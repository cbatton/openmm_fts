import time
from pathlib import Path

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
    """
    OMM Interface with a given system
    """

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
        plumed_force=None,
        comm=None,
    ):
        self.folder_name = folder_name
        self.base_filename = f"{self.folder_name}"
        print(f"Base filename: {self.base_filename}")
        # Get count from a count file if it exists, otherwise set to 0
        count_file = Path(f"{self.base_filename}_count.txt")
        if count_file.exists():
            count = int(np.loadtxt(count_file) + 1)
            np.savetxt(count_file, [count], fmt="%d")
        else:
            count = 0
            np.savetxt(count_file, [count], fmt="%d")
        self.count = count
        self.base_filename_2 = f"{self.folder_name}_{self.count}"
        self.internal_count = 0
        self.save_int = save_int
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
        self.force_groups = force_groups
        self.parameter_name = parameter_name
        self.parameter_force_name = parameter_force_name
        self.comm = comm
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        if (
            integrator_name == "csvr_leapfrog"
            or integrator_name == "csvr_leapfrog_end"
            or integrator_name == "csvr_verlet"
            or integrator_name == "csvr_verlet_end"
        ):
            # Remove csvr_ from integrator name, and set integrator
            integrator_name = integrator_name[5:]
            integrator = CSVRIntegrator(
                system=system.system,
                temperature=temperature,
                tau=friction,
                timestep=time_step,
                scheme=integrator_name,
            )
            integrator.setRandomNumberSeed(seed)
        elif integrator_name == "langevin":
            integrator = LangevinMiddleIntegrator(
                temperature,
                friction,
                time_step,
            )
            integrator.setRandomNumberSeed(seed)
        elif integrator_name == "brownian":
            integrator = BrownianIntegrator(
                temperature,
                friction,
                time_step,
            )
            integrator.setRandomNumberSeed(seed)
        else:
            raise ValueError("Integrator name not recognized")

        platform = Platform.getPlatformByName(platform)
        properties = {}
        if platform == "CUDA":
            properties = {"Precision": precision}
        elif platform == "CPU":
            properties = {"Threads": "1"}

        # Set up plumed if using
        if plumed_force is not None:
            system.system.addForce(plumed_force)

        self.simulation = self._init_simulation(
            system.system, integrator, platform, properties, system.topology
        )
        if count == 0:
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
        # Get number of atoms in the system
        self.num_atoms = len(system.positions)
        print(f"Number of atoms: {self.num_atoms}")
        reporter = HDF5Reporter(
            f"{self.base_filename_2}_{self.internal_count}.h5", self.save_int
        )
        restart_reporter = CheckpointReporter(
            f"{self.base_filename}_restart.chk", self.save_int
        )
        self.simulation.reporters.append(restart_reporter)
        self.simulation.reporters.append(reporter)
        # save initial state
        self.simulation.saveCheckpoint(f"{self.base_filename_2}_restart.chk")
        self.simulation.saveState(f"{self.base_filename_2}_restart.xml")
        # replica exchange info
        # should collect all information needed to do it
        self.beta = 1.0 / (temperature * BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA)
        self.beta = self.beta.value_in_unit_system(md_unit_system)
        if self.comm is not None:
            self.nswap_attemps = self.size**3
            self.parameter_values_rank = np.zeros(len(parameter_name), dtype=np.float64)
            self.force_values_rank = np.zeros(
                len(parameter_force_name), dtype=np.float64
            )
            for i, name in enumerate(parameter_name):
                self.parameter_values_rank[i] = self.simulation.context.getParameter(
                    name
                )
            for i, name in enumerate(parameter_force_name):
                self.force_values_rank[i] = self.simulation.context.getParameter(name)
            self.parameter_values = np.zeros(
                (self.size, len(parameter_name)), dtype=np.float64
            )
            self.force_values = np.zeros(
                (self.size, len(parameter_force_name)), dtype=np.float64
            )
            # use MPI to get all parameter values
            self.comm.Allgather(self.parameter_values_rank, self.parameter_values)
            self.comm.Allgather(self.force_values_rank, self.force_values)
            self.replica_rank = self.rank
            if self.rank == 0:
                print(f"Parameter values: {self.parameter_values}")
                print(f"Force values: {self.force_values}")
                print("Beta value:", self.beta)
            # should have something here to handle restarting
            # as not necessarily the case the initial values are what they should be

    def run_sim(self, steps, close_file=False):
        """Runs self.simulation for steps steps
        Arguments:
            steps: The number of steps to run the simulation for
            close_file: A bool to determine whether to close file. Necessary
            if using HDF5Reporter
        """
        self.simulation.step(steps)
        if close_file:
            self.simulation.reporters[1].close()

    def _init_simulation(self, system, integrator, platform, properties, topology):
        """Initializes an OpenMM simulation
        Arguments:
            system: An OpenMM system
            integrator: An OpenMM integrator
            platform: An OpenMM platform specifying the device information
            num_beads: An int specifying the number of beads to use
        Returns:
            simulation: An OpenMM simulation object
        """
        simulation = Simulation(topology, system, integrator, platform, properties)
        return simulation

    def get_information(self, as_numpy=True, enforce_periodic_box=True):
        """Gets information (positions, forces and PE of system)
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
        """Generates long trajectory of length num_data_points*save_freq time steps where information (pos, vel, forces, pe, ke, cell, cvs)
           are saved every save_freq time steps
        Arguments:
            init_pos: A numpy array of shape (n_atoms, 3) corresponding to the initial positions in Angstroms
            num_data_points: An int specifying the number of data points to generate
            save_freq: An int specifying the frequency to save data
            h5_freq: An int specifying the frequency to make a new h5 file
            enforce_periodic_box: A boolean of whether to enforce periodic boundary conditions
            tag: A string specifying the tag to save the data
            time_max: An int specifying the maximum time to run the simulation for
            precision: An int specifying the precision of the storage
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
                # get all cvs, then evaluate probability of swapping
                cvs_all = np.zeros((self.size, len(cvs)), dtype=np.float64)
                # use MPI to get all cvs
                self.comm.Allgather(cvs, cvs_all)
                # get current replica_rank
                replica_rank_all = np.zeros(self.size, dtype=np.int64)
                self.comm.Allgather(
                    np.array(self.replica_rank, dtype=np.int64), replica_rank_all
                )
                # swap
                if self.rank == 0:
                    replica_rank_new, num_accepted, num_attempted = mix_replicas(
                        self.beta,
                        cvs_all,
                        self.parameter_values,
                        self.force_values,
                        replica_rank_all,
                        self.nswap_attemps,
                    )
                    print(f"Replica rank: {replica_rank_new}")
                    print(f"Number of accepted swaps: {num_accepted}")
                    print(f"Number of attempted swaps: {num_attempted}")
                else:
                    replica_rank_new = np.zeros(self.size, dtype=np.int64)
                # communicate the new replica rank
                self.comm.Bcast(replica_rank_new, root=0)
                if replica_rank_new[self.rank] != self.replica_rank:
                    # update the context parameters
                    for i, name in enumerate(self.parameter_name):
                        self.simulation.context.setParameter(
                            name, self.parameter_values[replica_rank_new[self.rank], i]
                        )
                    for i, name in enumerate(self.parameter_force_name):
                        self.simulation.context.setParameter(
                            name, self.force_values[replica_rank_new[self.rank], i]
                        )
                self.replica_rank = replica_rank_new[self.rank]

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


@numba.njit
def mix_replicas(
    beta,
    cvs_all,
    parameter_values,
    force_values,
    replica_rank,
    nswap_attemps,
):
    """Mixes replicas based on the Metropolis criterion
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
    energy_ij = np.zeros((n_replicas, n_replicas), dtype=np.float64)
    for i in range(n_replicas):
        for j in range(n_replicas):
            if i != j:
                energy_ij[i, j] = 0.5 * np.sum(
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

        energy_ii = 0.5 * np.sum(
            force_values[state_i] * (cvs_all[i] - parameter_values[state_i]) ** 2
        )
        energy_jj = 0.5 * np.sum(
            force_values[state_j] * (cvs_all[j] - parameter_values[state_j]) ** 2
        )
        energy_ij = 0.5 * np.sum(
            force_values[state_i] * (cvs_all[j] - parameter_values[state_i]) ** 2
        )
        energy_ji = 0.5 * np.sum(
            force_values[state_j] * (cvs_all[i] - parameter_values[state_j]) ** 2
        )

        log_p_accept = -beta * (energy_ij + energy_ji - energy_ii - energy_jj)

        # Calculate the acceptance probability
        # Decide whether to swap based on the acceptance probability
        if log_p_accept >= 0 or np.random.rand() < np.exp(log_p_accept):
            # Swap the replicas
            replica_rank[i] = state_j
            replica_rank[j] = state_i
            num_accepted[i, j] += 1
            num_accepted[j, i] += 1

    return replica_rank, num_accepted, num_attempted
