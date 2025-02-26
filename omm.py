import time
from pathlib import Path

import h5py
import numpy as np
from mdtraj.reporters import HDF5Reporter
from openmm import (
    BrownianIntegrator,
    CMMotionRemover,
    LangevinMiddleIntegrator,
    Platform,
)
from openmm.app import CheckpointReporter, Simulation
from openmm.unit import kelvin, md_unit_system, nanometers, picoseconds
from openmm_csvr.csvr import CSVRIntegrator
from scipy.interpolate import CubicSpline

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
        string_freq=None,
        string_dt=None,
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
        self.string_freq = string_freq
        self.string_dt = string_dt
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
        # get masses of the atoms
        self.masses = np.array(
            [
                system.system.getParticleMass(i).value_in_unit_system(md_unit_system)
                for i in range(self.num_atoms)
            ]
        )

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
            forces_metric = []
            for i, custom_force in enumerate(self.string_forces):
                state = self.simulation.context.getState(
                    getEnergy=True,
                    getForces=True,
                    groups={self.force_groups[i]},
                )
                cvs.append(
                    state.getPotentialEnergy().value_in_unit_system(md_unit_system)
                )
                forces_metric.append(
                    state.getForces(asNumpy=as_numpy).value_in_unit_system(
                        md_unit_system
                    )
                )
            forces_metric = np.array(forces_metric)
            metric = np.zeros((len(cvs), len(cvs)), dtype=np.float64)
            for i in range(len(cvs)):
                for j in range(i, len(cvs)):
                    metric[i, j] = np.sum(
                        forces_metric[i]
                        * forces_metric[j]
                        / (self.masses[:, np.newaxis])
                    )
                    metric[j, i] = metric[i, j]
            self.cvs_store.append(cvs)
            self.metric_store.append(metric)
            # turn off track
            self.simulation.context.setParameter("track", 0)

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

        # prepare h5 file on rank 0 to store string data at each update
        h5_string = h5py.File(f"{self.base_filename_2}_string.h5", "w")
        string_count = 0

        h5_chunk = -(start_iter % h5_freq) + h5_freq + 1
        h5_file = TrajWriter(
            f"{self.base_filename_2}_{self.internal_count}_data.h5",
            self.num_atoms,
            h5_chunk,
            precision=precision,
            cvs=self.string_forces,
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

            h5_file.write_frame(positions, velocities, forces, pe, ke, cell, cvs)

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
                )

            if self.string_freq is not None:
                if (
                    _ % self.string_freq == 0
                    and _ != 0
                    and _ / self.string_freq > burn_in
                ):
                    # get initial parameters
                    cvs0 = []
                    for i in range(len(self.parameter_name)):
                        cvs0.append(
                            self.simulation.context.getParameter(self.parameter_name[i])
                        )
                    cvs0 = np.array(cvs0)
                    # average the CVs, metric
                    # as periodic, do
                    cvs_store_np = np.array(self.cvs_store)
                    # subtract the initial cvs
                    cvs_store_np -= cvs0
                    cvs_store_np = cvs_store_np - 2 * np.pi * np.rint(
                        cvs_store_np / (2 * np.pi)
                    )
                    # mean_cvs_x = np.mean(np.cos(cvs_store_np), axis=0)
                    # mean_cvs_y = np.mean(np.sin(cvs_store_np), axis=0)
                    # cvs = np.arctan2(mean_cvs_y, mean_cvs_x)
                    cvs_diff = np.mean(cvs_store_np, axis=0)
                    metric = np.mean(self.metric_store, axis=0)
                    cv_update = cvs0 + self.string_dt * metric @ cvs_diff
                    cv_update = cv_update - 2 * np.pi * np.rint(cv_update / (2 * np.pi))
                    # String method communication steps
                    if self.comm is not None:
                        cv_global = np.zeros((self.size, len(cv_update)))
                        self.comm.Allgather(cv_update, cv_global)
                        if self.rank == 0:
                            # interpolate the string, and update the parameters
                            t = np.linspace(0, 1, self.size)
                            cv_global = np.unwrap(cv_global, axis=0)
                            cs_spline = CubicSpline(t, cv_global, axis=0)
                            # choose new t-values such that the string is equally spaced in CV-space
                            arc_length = np.zeros(self.size)
                            diffs = np.diff(cv_global, axis=0)
                            segment_lengths = np.linalg.norm(diffs, axis=1)
                            arc_length[1:] = np.cumsum(segment_lengths)
                            arc_length /= arc_length[-1]
                            equal_spaced_arc_lengths = np.linspace(0, 1, self.size)
                            t_equal = np.interp(equal_spaced_arc_lengths, arc_length, t)
                            cv_global = cs_spline(t_equal)
                            # convert back to being in the range of -pi to pi
                            cv_global = cv_global - 2 * np.pi * np.rint(
                                cv_global / (2 * np.pi)
                            )
                            print(cv_global)
                            h5_string.create_group(f"config_{string_count}")
                            h5_string[f"config_{string_count}"].create_dataset(
                                "cvs",
                                data=cv_global,
                                dtype=np.float64,
                            )
                            string_count += 1
                            self.comm.Scatter(cv_global, cv_update, root=0)
                        else:
                            self.comm.Scatter(cv_global, cv_update, root=0)

                    for i in range(len(self.parameter_name)):
                        self.simulation.context.setParameter(
                            self.parameter_name[i], cv_update[i]
                        )
                    self.cvs_store = []
                    self.metric_store = []
                elif _ % self.string_freq == 0 and _ / self.string_freq <= burn_in:
                    self.cvs_store = []
                    self.metric_store = []

            # End timer
            time_end = time.time()
            # If time is greater than time_max, break
            if time_end - time_start > time_max:
                self.simulation.reporters[1].close()
                h5_file.early_close()
                h5_string.close()
                exit(0)
        # If naturally ended, close the h5 files
        self.simulation.reporters[1].close()
        h5_file.early_close()
        h5_string.close()
