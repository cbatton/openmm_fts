"""OpenMM interface for doing Hamiltonian replica exchange simulations."""

import time
from pathlib import Path

import h5py
import numba
import numpy as np
from mdtraj.reporters import HDF5Reporter
from openmm import (
    BrownianIntegrator,
    CMMotionRemover,
    LangevinMiddleIntegrator,
    Platform,
)
from openmm.app import CheckpointReporter, Simulation
from openmm.unit import (
    AVOGADRO_CONSTANT_NA,
    BOLTZMANN_CONSTANT_kB,
    kelvin,
    md_unit_system,
    nanometers,
    picoseconds,
)
from openmm_csvr.csvr import CSVRIntegrator

from traj_writer import TrajWriter


class OMMFF:
    """OMM Interface with a given system."""

    def __init__(
        self,
        system,
        platform="CUDA",
        precision="single",
        integrator_name="csvr_leapfrog",
        temperature=300.0 * kelvin,
        velocities_com=True,
        time_step=0.001 * picoseconds,
        friction=0.1 / picoseconds,
        seed=1,
        save_int=10,
        folder_name="",
        custom_forces=None,
        string_forces=None,
        force_groups=None,
        parameter_name=None,
        parameter_force_name=None,
        comm=None,
        swap_scheme="mixing",
    ):
        self._setup_filenames_and_count(folder_name)
        self._initialize_system_forces(
            system,
            velocities_com,
            custom_forces,
            string_forces,
        )
        self.force_groups = force_groups
        self.parameter_name = parameter_name
        self.parameter_force_name = parameter_force_name
        self.comm = comm
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

        integrator = self._create_integrator(
            integrator_name,
            temperature,
            friction,
            time_step,
            seed,
        )
        self._initialize_simulation(
            system,
            integrator,
            platform,
            precision,
            system.topology,
        )
        self._load_or_initialize(system, temperature)
        self._setup_reporters(save_int)
        self._setup_swap_scheme(swap_scheme, temperature)
        self.num_atoms = len(system.positions)

    def _setup_filenames_and_count(self, folder_name):
        """Sets up the filenames and count for the simulation."""
        self.folder_name = folder_name
        self.base_filename = f"{self.folder_name}"
        count_file = Path(f"{self.base_filename}_count.txt")
        if count_file.exists():
            self.count = int(np.loadtxt(count_file) + 1)
        else:
            self.count = 0
        np.savetxt(count_file, [self.count], fmt="%d")
        self.base_filename_2 = f"{self.folder_name}_{self.count}"
        self.internal_count = 0

    def _initialize_system_forces(
        self, system, velocities_com, custom_forces, string_forces
    ):
        """Initializes the system forces for the simulation."""
        if velocities_com:
            system.system.addForce(CMMotionRemover())
        if custom_forces is not None:
            if not isinstance(custom_forces, list):
                custom_forces = [custom_forces]
            for custom_force in custom_forces:
                system.system.addForce(custom_force)
        self.custom_forces = custom_forces
        if string_forces is not None:
            if not isinstance(string_forces, list):
                string_forces = [string_forces]
            for custom_force in string_forces:
                system.system.addForce(custom_force)
            self.cvs_store = []
            self.metric_store = []
        self.string_forces = string_forces

    def _create_integrator(
        self, integrator_name, temperature, friction, time_step, seed
    ):
        """Creates an OpenMM integrator based on the given parameters."""
        if integrator_name == "csvr_leapfrog":
            integrator = CSVRIntegrator(
                system=self.simulation.system,
                temperature=temperature,
                tau=friction,
                timestep=time_step,
                scheme="leapfrog",
            )
        elif integrator_name == "csvr_leapfrog_end":
            integrator = CSVRIntegrator(
                system=self.simulation.system,
                temperature=temperature,
                tau=friction,
                timestep=time_step,
                scheme="leapfrog_end",
            )
        elif integrator_name == "csvr_verlet":
            integrator = CSVRIntegrator(
                system=self.simulation.system,
                temperature=temperature,
                tau=friction,
                timestep=time_step,
                scheme="verlet",
            )
        elif integrator_name == "csvr_verlet_end":
            integrator = CSVRIntegrator(
                system=self.simulation.system,
                temperature=temperature,
                tau=friction,
                timestep=time_step,
                scheme="verlet_end",
            )
        elif integrator_name == "langevin":
            integrator = LangevinMiddleIntegrator(
                temperature,
                friction,
                time_step,
            )
        elif integrator_name == "brownian":
            integrator = BrownianIntegrator(
                temperature,
                friction,
                time_step,
            )
        else:
            raise ValueError("Integrator name not recognized")
        integrator.setRandomNumberSeed(seed)
        return integrator

    def _initialize_simulation(self, system, integrator, platform, precision, topology):
        """Initializes an OpenMM simulation.

        Arguments:
            system: An OpenMM system
            integrator: An OpenMM integrator
            platform: An OpenMM platform specifying the device information
            precision: A string specifying the precision of the simulation
            topology: An OpenMM topology object
        Returns:
            simulation: An OpenMM simulation object
        """
        platform = Platform.getPlatformByName(platform)
        properties = {}
        if platform == "CUDA":
            properties = {"Precision": precision}
        elif platform == "CPU":
            properties = {"Threads": "1"}

        self.simulation = self._init_simulation(
            system,
            integrator,
            platform,
            properties,
            topology,
        )

    def _init_simulation(self, system, integrator, platform, properties, topology):
        """Initializes an OpenMM simulation.

        Arguments:
            system: An OpenMM system
            integrator: An OpenMM integrator
            platform: An OpenMM platform specifying the device information
            properties: A dictionary of properties for the platform
            topology: An OpenMM topology object
        Returns:
            simulation: An OpenMM simulation object
        """
        simulation = Simulation(topology, system, integrator, platform, properties)
        return simulation

    def _load_or_initialize(self, system, temperature):
        """Loads or initializes the simulation with a given system and temperature.

        Arguments:
            system: An OpenMM system
            temperature: A float specifying the temperature in Kelvin
        """
        if self.count == 0:
            print("Minimizing energy")
            self.simulation.context.setPositions(system.positions)
            self.simulation.minimizeEnergy()
            self.simulation.context.setVelocitiesToTemperature(temperature)
        else:
            try:
                self.simulation.loadCheckpoint(f"{self.base_filename}_restart.chk")
            except Exception:
                # Previous one failed, try loading old restart file
                base_filename_old = f"{self.folder_name}_{self.count - 1}"
                self.simulation.loadCheckpoint(f"{base_filename_old}_restart.chk")
        self.simulation.saveCheckpoint(f"{self.base_filename_2}_restart.chk")
        self.simulation.saveState(f"{self.base_filename_2}_restart.xml")

    def _setup_reporters(self, save_int):
        """Sets up the reporters for the simulation."""
        self.save_int = save_int
        # Set up reporters
        self.simulation.reporters = []
        if save_int > 0:
            reporter = HDF5Reporter(
                f"{self.base_filename_2}_{self.internal_count}.h5", save_int
            )
            restart_reporter = CheckpointReporter(
                f"{self.base_filename}_restart.chk", save_int
            )
            self.simulation.reporters.append(reporter)
            self.simulation.reporters.append(restart_reporter)

    def _setup_swap_scheme(self, swap_scheme, temperature):
        """Sets up the swap scheme for the simulation."""
        if swap_scheme not in ["mixing", "neighbors"]:
            raise ValueError(
                f"swap_scheme must be 'mixing' or 'neighbors', got '{swap_scheme}'"
            )
        self.swap_scheme = swap_scheme
        self.beta = 1.0 / (temperature * BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA)
        self.beta = self.beta.value_in_unit_system(md_unit_system)
        if self.comm is not None:
            self.nswap_attemps = self.size**3
            self.parameter_values_rank = np.zeros(
                len(self.parameter_name), dtype=np.float64
            )
            self.force_values_rank = np.zeros(
                len(self.parameter_force_name), dtype=np.float64
            )
            for i, name in enumerate(self.parameter_name):
                self.parameter_values_rank[i] = self.simulation.context.getParameter(
                    name
                )
            for i, name in enumerate(self.parameter_force_name):
                self.force_values_rank[i] = self.simulation.context.getParameter(name)
            self.parameter_values = np.zeros(
                (self.size, len(self.parameter_name)), dtype=np.float64
            )
            self.force_values = np.zeros(
                (self.size, len(self.parameter_force_name)), dtype=np.float64
            )
            # use MPI to get all parameter values
            self.comm.Allgather(self.parameter_values_rank, self.parameter_values)
            self.comm.Allgather(self.force_values_rank, self.force_values)
            if Path("replica_ranks.h5").exists():
                with h5py.File("replica_ranks.h5", "r") as f:
                    # get the last config
                    replica_rank_new = f[list(f.keys())[-1]][:]
                    self.replica_rank = replica_rank_new[self.rank]
                    for i, name in enumerate(self.parameter_name):
                        self.simulation.context.setParameter(
                            name, self.parameter_values[self.replica_rank, i]
                        )
                    for i, name in enumerate(self.parameter_force_name):
                        self.simulation.context.setParameter(
                            name, self.force_values[self.replica_rank, i]
                        )

    def run_sim(self, steps, close_file=False):
        """Runs self.simulation for given number of steps.

        Arguments:
            steps: The number of steps to run the simulation for
            close_file: A bool to determine whether to close file. Necessary
            if using HDF5Reporter
        """
        self.simulation.step(steps)
        if close_file:
            self.simulation.reporters[1].close()

    def get_information(self, as_numpy=True, enforce_periodic_box=True):
        """Gets information (positions, forces and PE of system).

        Arguments:
            as_numpy: A boolean of whether to return as a numpy array
            enforce_periodic_box: A boolean of whether to enforce periodic boundary conditions
        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in nm
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in nm/ps
            forces: A numpy array of shape (n_atoms, 3) corresponding to the force in kJ/mol*nm
            pe: A float coressponding to the potential energy in kJ/mol
            ke: A float coressponding to the kinetic energy in kJ/mol
            cell: A numpy array of shape (3, 3) corresponding to the cell vectors in nm
            cvs: A list of numpy arrays of shape (n_cvs,) corresponding to the collective variables
        """
        state = self.simulation.context.getState(
            getForces=True,
            getEnergy=True,
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=enforce_periodic_box,
        )
        positions = state.getPositions(asNumpy=as_numpy).value_in_unit_system(
            md_unit_system
        )
        forces = state.getForces(asNumpy=as_numpy).value_in_unit_system(md_unit_system)
        velocities = state.getVelocities(asNumpy=as_numpy).value_in_unit_system(
            md_unit_system
        )

        pe = state.getPotentialEnergy().value_in_unit_system(md_unit_system)
        ke = state.getKineticEnergy().value_in_unit_system(md_unit_system)
        cell = state.getPeriodicBoxVectors(asNumpy=as_numpy).value_in_unit_system(
            md_unit_system
        )

        cvs = []
        if self.string_forces is not None:
            # turn on track
            self.simulation.context.setParameter("track", 1)
            for i in range(len(self.string_forces)):
                state = self.simulation.context.getState(
                    getEnergy=True,
                    groups={self.force_groups[i]},
                )
                cvs.append(
                    state.getPotentialEnergy().value_in_unit_system(md_unit_system)
                )

            self.cvs_store.append(cvs)
            # turn off track
            self.simulation.context.setParameter("track", 0)
        cvs = np.array(cvs, dtype=np.float64)

        return positions, velocities, forces, pe, ke, cell, cvs

    def generate_long_trajectory(
        self,
        init_pos=None,
        num_data_points=int(1e8),
        save_freq=1000,
        h5_freq=100,
        enforce_periodic_box=True,
        tag=None,
        time_max=117 * 60,
        precision=32,
        burn_in=1,
        swap_freq=1,
    ):
        """Generates long trajectory of length num_data_points*save_freq time steps where information (pos, vel, forces, pe, ke, cell, cvs) are saved every save_freq time steps.

        Arguments:
            init_pos: A numpy array of shape (n_atoms, 3) corresponding to the initial positions in Angstroms
            num_data_points: An int specifying the number of data points to generate
            save_freq: An int specifying the frequency to save data
            h5_freq: An int specifying the frequency to make a new h5 file
            enforce_periodic_box: A boolean of whether to enforce periodic boundary conditions
            tag: A string specifying the tag to save the data
            time_max: An int specifying the maximum time to run the simulation for
            precision: An int specifying the precision of the storage
            burn_in: An int specifying the number of initial steps to burn in before swapping
            swap_freq: An int specifying the frequency to swap replicas
        """
        if tag is None:
            tag = self.base_filename

        if init_pos is not None:
            init_pos = init_pos * nanometers
            self.simulation.context.setPositions(init_pos)

        # Start a timer
        time_start = time.time()
        start_iter = 0
        try:
            data = np.loadtxt(tag + "_pe.txt")
            start_iter = len(data)
        except Exception:
            start_iter = 0
            pass

        h5_chunk = -(start_iter % h5_freq) + h5_freq + 1
        h5_file = TrajWriter(
            f"{self.base_filename_2}_{self.internal_count}_data.h5",
            self.num_atoms,
            h5_chunk,
            precision=precision,
            cvs=self.string_forces,
            rank=True,
        )
        for _ in range(start_iter, num_data_points):
            self.run_sim(save_freq)
            (
                positions,
                velocities,
                forces,
                pe,
                ke,
                cell,
                cvs,
            ) = self.get_information()

            # save plainly pe to keep a count
            f = open(f"{tag}_pe.txt", "ab")
            np.savetxt(f, np.expand_dims(pe, 0), fmt="%.9e")
            f.close()

            if _ % swap_freq == 0 and self.comm is not None and _ != 0:
                self._perform_replica_swap(cvs)

            h5_file.write_frame(
                positions,
                velocities,
                forces,
                pe,
                ke,
                cell,
                cvs=cvs,
                rank=self.replica_rank,
            )

            if _ % h5_freq == 0 and _ != 0:
                self.simulation.reporters[1].close()
                self.internal_count += 1
                self.simulation.reporters[1] = HDF5Reporter(
                    f"{self.base_filename_2}_{self.internal_count}.h5",
                    self.save_int,
                )
                h5_file.close()
                h5_file = TrajWriter(
                    f"{self.base_filename_2}_{self.internal_count}_data.h5",
                    self.num_atoms,
                    h5_freq,
                    precision=precision,
                    cvs=self.string_forces,
                    rank=True,
                )

            # End timer
            time_end = time.time()
            # If time is greater than time_max, break
            if time_end - time_start > time_max:
                self.simulation.reporters[1].close()
                h5_file.early_close()
                exit(0)
        # If naturally ended, close the h5 files
        self.simulation.reporters[1].close()
        h5_file.early_close()

    def _gather_replica_data(self, cvs):
        """Gather collective variables and replica ranks from all processes."""
        cvs_all = np.zeros((self.size, len(cvs)), dtype=np.float64)
        self.comm.Allgather(cvs, cvs_all)

        replica_rank_all = np.zeros(self.size, dtype=np.int64)
        self.comm.Allgather(
            np.array(self.replica_rank, dtype=np.int64), replica_rank_all
        )

        return cvs_all, replica_rank_all

    def _update_replica_parameters(self, replica_rank_new):
        """Update simulation context parameters based on new replica assignment."""
        if replica_rank_new[self.rank] != self.replica_rank:
            for i, name in enumerate(self.parameter_name):
                self.simulation.context.setParameter(
                    name, self.parameter_values[replica_rank_new[self.rank], i]
                )
            for i, name in enumerate(self.parameter_force_name):
                self.simulation.context.setParameter(
                    name, self.force_values[replica_rank_new[self.rank], i]
                )
        self.replica_rank = replica_rank_new[self.rank]

    def _save_swap_results(self, replica_rank_new):
        """Save replica ranks and swap rates to HDF5 files."""
        with h5py.File("replica_ranks.h5", "a") as f:
            if len(f.keys()) == 0:
                f.create_dataset(
                    f"config_{self.count}_{len(f.keys())}",
                    data=replica_rank_new,
                )
            else:
                dset_name = f"config_{self.count}_{len(f.keys())}"
                f.create_dataset(dset_name, data=replica_rank_new)

        with h5py.File("swap_rates.h5", "a") as f:
            if len(f.keys()) == 0:
                group = f.create_group(f"config_{self.count}_{len(f.keys())}")
                group.create_dataset("num_accepted", data=self.num_accepted)
                group.create_dataset("num_attempted", data=self.num_attempted)
            else:
                group = f.create_group(f"config_{self.count}_{len(f.keys())}")
                group.create_dataset("num_accepted", data=self.num_accepted)
                group.create_dataset("num_attempted", data=self.num_attempted)

    def _perform_replica_swap(self, cvs):
        """Perform replica exchange swap attempt."""
        cvs_all, replica_rank_all = self._gather_replica_data(cvs)

        if self.rank == 0:
            if self.swap_scheme == "neighbors":
                replica_rank_new, num_accepted, num_attempted = (
                    mix_neighboring_replicas(
                        self.beta,
                        cvs_all,
                        self.parameter_values,
                        self.force_values,
                        replica_rank_all,
                    )
                )
            elif self.swap_scheme == "mixing":
                replica_rank_new, num_accepted, num_attempted = mix_replicas(
                    self.beta,
                    cvs_all,
                    self.parameter_values,
                    self.force_values,
                    replica_rank_all,
                    self.nswap_attemps,
                )

            if self.num_attempted is None:
                self.num_attempted = num_attempted
                self.num_accepted = num_accepted
            else:
                self.num_attempted += num_attempted
                self.num_accepted += num_accepted

            print(f"Replica rank: {replica_rank_new}")
            print(f"Number of accepted swaps: {self.num_accepted}")
            print(f"Number of attempted swaps: {self.num_attempted}")
        else:
            replica_rank_new = np.zeros(self.size, dtype=np.int64)

        # Broadcast the new replica rank to all processes
        self.comm.Bcast(replica_rank_new, root=0)

        # Update parameters and save results
        self._update_replica_parameters(replica_rank_new)

        if self.rank == 0:
            self._save_swap_results(replica_rank_new)


@numba.njit
def mix_replicas(
    beta,
    cvs_all,
    parameter_values,
    force_values,
    replica_rank,
    nswap_attemps,
):
    """Mixes replicas based on the Metropolis criterion.

    Arguments:
        beta: A float corresponding to the inverse temperature
        cvs_all: A numpy array of shape (n_replicas, n_cvs) corresponding to the collective variables of all replicas
        parameter_values: A numpy array of shape (n_replicas, n_parameters) corresponding to the parameter values of all replicas
        force_values: A numpy array of shape (n_replicas, n_forces) corresponding to the force values of all replicas
        replica_rank: An int corresponding to the rank of the current replica
        nswap_attemps: An int specifying the number of swap attempts
    Returns:
        replica_rank: An int corresponding to the new rank of the replica after mixing
        num_accepted: A numpy array of shape (n_replicas, n_replicas) corresponding to the number of accepted swaps between replicas
        num_attempted: A numpy array of shape (n_replicas, n_replicas) corresponding to the number of attempted swaps between replicas
    """
    # precompute the acceptance probabilities
    n_replicas = cvs_all.shape[0]
    energy_ij_store = np.zeros((n_replicas, n_replicas), dtype=np.float64)
    for i in range(n_replicas):
        for j in range(n_replicas):
            energy_ij_store[i, j] = 0.5 * np.sum(
                force_values[i] * (cvs_all[j] - parameter_values[i]) ** 2
            )

    # Initialize the number of accepted and attempted swaps
    num_accepted = np.zeros((n_replicas, n_replicas), dtype=np.int64)
    num_attempted = np.zeros((n_replicas, n_replicas), dtype=np.int64)

    for _ in range(nswap_attemps):
        # Randomly select two replicas to swap
        i = np.random.randint(n_replicas)
        j = np.random.randint(n_replicas)
        while i == j:
            j = np.random.randint(n_replicas)

        state_i = replica_rank[i]
        state_j = replica_rank[j]

        # Increment the attempted swaps
        num_attempted[state_i, state_j] += 1
        num_attempted[state_j, state_i] += 1

        energy_ii = energy_ij_store[state_i, i]
        energy_jj = energy_ij_store[state_j, j]
        energy_ij = energy_ij_store[state_i, j]
        energy_ji = energy_ij_store[state_j, i]

        log_p_accept = -beta * (energy_ij + energy_ji - energy_ii - energy_jj)

        # Calculate the acceptance probability
        # Decide whether to swap based on the acceptance probability
        if log_p_accept >= 0 or np.random.rand() < np.exp(log_p_accept):
            # Swap the replicas
            replica_rank[i] = state_j
            replica_rank[j] = state_i
            num_accepted[state_i, state_j] += 1
            num_accepted[state_j, state_i] += 1

    return replica_rank, num_accepted, num_attempted


def mix_neighboring_replicas(
    beta,
    cvs_all,
    parameter_values,
    force_values,
    replica_rank,
):
    """Mixes replicas based on the Metropolis criterion.

    Arguments:
        beta: A float corresponding to the inverse temperature
        cvs_all: A numpy array of shape (n_replicas, n_cvs) corresponding to the collective variables of all replicas
        parameter_values: A numpy array of shape (n_replicas, n_parameters) corresponding to the parameter values of all replicas
        force_values: A numpy array of shape (n_replicas, n_forces) corresponding to the force values of all replicas
        replica_rank: An int corresponding to the rank of the current replica
        nswap_attemps: An int specifying the number of swap attempts
    Returns:
        replica_rank: An int corresponding to the new rank of the replica after mixing
        num_accepted: A numpy array of shape (n_replicas, n_replicas) corresponding to the number of accepted swaps between replicas
        num_attempted: A numpy array of shape (n_replicas, n_replicas) corresponding to the number of attempted swaps between replicas
    """
    # precompute the acceptance probabilities
    n_replicas = cvs_all.shape[0]
    energy_ij_store = np.zeros((n_replicas, n_replicas), dtype=np.float64)
    for i in range(n_replicas):
        for j in range(n_replicas):
            energy_ij_store[i, j] = 0.5 * np.sum(
                force_values[i] * (cvs_all[j] - parameter_values[i]) ** 2
            )

    # Initialize the number of accepted and attempted swaps
    num_accepted = np.zeros((n_replicas, n_replicas), dtype=np.int64)
    num_attempted = np.zeros((n_replicas, n_replicas), dtype=np.int64)

    # Randomly select pairs of neighboring replicas to swap
    offset = np.random.randint(2)
    for state_i in range(offset, n_replicas - 1, 2):
        state_j = state_i + 1
        i = np.where(replica_rank == state_i)[0][0]
        j = np.where(replica_rank == state_j)[0][0]

        # Increment the attempted swaps
        num_attempted[state_i, state_j] += 1
        num_attempted[state_j, state_i] += 1

        energy_ii = energy_ij_store[state_i, i]
        energy_jj = energy_ij_store[state_j, j]
        energy_ij = energy_ij_store[state_i, j]
        energy_ji = energy_ij_store[state_j, i]

        log_p_accept = -beta * (energy_ij + energy_ji - energy_ii - energy_jj)

        # Calculate the acceptance probability
        # Decide whether to swap based on the acceptance probability
        if log_p_accept >= 0 or np.random.rand() < np.exp(log_p_accept):
            # Swap the replicas
            replica_rank[i] = state_j
            replica_rank[j] = state_i
            num_accepted[state_i, state_j] += 1
            num_accepted[state_j, state_i] += 1

    return replica_rank, num_accepted, num_attempted
