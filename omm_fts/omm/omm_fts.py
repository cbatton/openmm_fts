"""OpenMM interface for the string method with collective variables."""

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
    VerletIntegrator,
)
from openmm.app import CheckpointReporter, Simulation
from openmm.unit import kelvin, kilojoule, md_unit_system, mole, nanometers, picoseconds
from openmm_csvr.csvr import CSVRIntegrator
from scipy.interpolate import interp1d

from ..io.traj_writer import TrajWriter


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
        string_freq=None,
        string_dt=None,
        string_kappa=None,
        cv_weights=None,
        update_ends=True,
        comm=None,
        minimize_init=True,
        minimize_intervals=None,
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
        self._initialize_string_parameters(
            string_kappa,
            string_freq,
            string_dt,
            cv_weights,
            update_ends,
            comm,
        )
        integrator = self._create_integrator(
            system,
            integrator_name,
            temperature,
            friction,
            time_step,
            seed,
        )

        # get number of atoms, masses of the atoms
        self.num_atoms = len(system.positions)
        self.masses = np.array([
            system.system.getParticleMass(i).value_in_unit_system(md_unit_system)
            for i in range(self.num_atoms)
        ])
        self._initialize_simulation(
            system,
            integrator,
            platform,
            precision,
            system.topology,
        )
        self._load_or_initialize(system, temperature, minimize_init, minimize_intervals)
        self._setup_reporters(save_int)

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

    def _initialize_string_parameters(
        self, string_kappa, string_freq, string_dt, cv_weights, update_ends, comm
    ):
        """Initializes the string parameters for the simulation."""
        self.string_kappa = string_kappa
        self.string_freq = string_freq
        self.string_dt = string_dt
        if cv_weights is not None:
            self.cv_weights = np.array(cv_weights)
        else:
            self.cv_weights = None
        self.update_ends = update_ends
        self.comm = comm
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        # create vectors for Thomas algorithm
        self.a_vec = np.zeros(self.size - 1, dtype=np.float64)
        self.b_vec = np.zeros(self.size, dtype=np.float64)
        self.c_vec = np.zeros(self.size - 1, dtype=np.float64)
        if self.string_kappa is not None:
            self.a_vec[:-1] = -self.string_kappa
            self.c_vec[1:] = -self.string_kappa
            self.b_vec[1:-1] = 1 + 2 * self.string_kappa
            self.b_vec[0] = 1
            self.b_vec[-1] = 1

    def _create_integrator(
        self, system, integrator_name, temperature, friction, time_step, seed
    ):
        """Creates an OpenMM integrator based on the given parameters."""
        if integrator_name == "csvr_leapfrog":
            integrator = CSVRIntegrator(
                system=system.system,
                temperature=temperature,
                tau=friction,
                timestep=time_step,
                scheme="leapfrog",
            )
        elif integrator_name == "csvr_leapfrog_end":
            integrator = CSVRIntegrator(
                system=system.system,
                temperature=temperature,
                tau=friction,
                timestep=time_step,
                scheme="leapfrog_end",
            )
        elif integrator_name == "csvr_verlet":
            integrator = CSVRIntegrator(
                system=system.system,
                temperature=temperature,
                tau=friction,
                timestep=time_step,
                scheme="verlet",
            )
        elif integrator_name == "csvr_verlet_end":
            integrator = CSVRIntegrator(
                system=system.system,
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
        elif integrator_name == "verlet":
            integrator = VerletIntegrator(time_step)
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
            system.system,
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

    def _load_or_initialize(
        self, system, temperature, minimize_init, minimize_intervals
    ):
        """Loads or initializes the simulation with a given system and temperature.

        Arguments:
            system: An OpenMM system
            temperature: A float specifying the temperature in Kelvin
            minimize_init: A boolean indicating whether to slowly initialize or not
            minimize_intervals: A boolean indicating the spacing used for minimization
        """
        if self.count == 0:
            print("Minimizing energy")
            self.simulation.context.setPositions(system.positions)
            if minimize_init:
                minimize_intervals = int(minimize_intervals)
                # slowly initialize
                # Get currents cvs0, see where they need to be, and then get there in minimize_intervals number of steps
                _, _, _, _, _, _, cvs0 = self.get_information()
                cvs0_target = []
                for i in range(len(self.parameter_name)):
                    cvs0_target.append(
                        self.simulation.context.getParameter(self.parameter_name[i])
                    )
                cvs0_target = np.array(cvs0_target)
                print(
                    f"Before minimization on {self.rank}: current CV values: {cvs0}, target CV values: {cvs0_target}"
                )
                cvs0_spline = np.zeros((minimize_intervals + 1, len(cvs0)))
                for i in range(len(cvs0)):
                    cvs0_spline[:, i] = np.linspace(
                        cvs0[i], cvs0_target[i], minimize_intervals + 1
                    )
                for i in range(minimize_intervals):
                    cvs0 = cvs0_spline[i]
                    for j in range(len(self.parameter_name)):
                        self.simulation.context.setParameter(
                            self.parameter_name[j], cvs0[j]
                        )
                    self.simulation.minimizeEnergy(
                        tolerance=0.1 * kilojoule / mole / nanometers
                    )
                _, _, _, _, _, _, cvs0 = self.get_information()
                print(
                    f"After minimization on {self.rank}: current CV values: {cvs0}, target CV values: {cvs0_target}"
                )
            else:
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
            self.simulation.reporters.append(restart_reporter)
            self.simulation.reporters.append(reporter)

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
            forces_metric = []
            for i, _custom_force in enumerate(self.string_forces):
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
        beta1=0.9,
        beta2=0.999,
    ):
        """Generates long trajectory with improved modularity."""
        # Setup phase
        tag = self._setup_trajectory_tag(tag)
        start_iter = self._setup_trajectory_start(tag)
        time_start = time.time()

        # Initialize trajectory files and Adam optimizer
        h5_string, h5_file = self._setup_trajectory_files(
            start_iter, h5_freq, precision, beta1, beta2
        )

        # Set initial positions if provided
        self._set_initial_positions(init_pos)

        # Main simulation loop
        string_count = 0
        for iteration in range(start_iter, num_data_points):
            if self._should_exit_early(time_start, time_max):
                self._cleanup_trajectory_files(h5_file, h5_string)
                self.comm.Barrier()
                return

            string_count, h5_file = self._run_trajectory_iteration(
                iteration,
                save_freq,
                tag,
                h5_file,
                h5_freq,
                precision,
                burn_in,
                string_count,
                h5_string,
            )

        # Natural completion cleanup
        self._cleanup_trajectory_files(h5_file, h5_string)

    def _setup_trajectory_tag(self, tag):
        """Setup trajectory tag."""
        return tag if tag is not None else self.base_filename

    def _setup_trajectory_start(self, tag):
        """Setup starting iteration from existing data."""
        try:
            data = np.loadtxt(tag + "_pe.txt")
            return len(data)
        except Exception:
            return 0

    def _setup_trajectory_files(self, start_iter, h5_freq, precision, beta1, beta2):
        """Setup trajectory output files."""
        # String file setup
        h5_string = h5py.File(f"{self.base_filename_2}_string.h5", "w")

        # Data file setup
        h5_chunk = -(start_iter % h5_freq) + h5_freq + 1
        h5_file = TrajWriter(
            f"{self.base_filename_2}_{self.internal_count}_data.h5",
            self.num_atoms,
            h5_chunk,
            precision=precision,
            cvs=self.string_forces,
        )

        # Adam optimizer setup
        self.adam = Adam(lr=self.string_dt, beta1=beta1, beta2=beta2)

        return h5_string, h5_file

    def _set_initial_positions(self, init_pos):
        """Set initial positions if provided."""
        if init_pos is not None:
            init_pos = init_pos * nanometers
            self.simulation.context.setPositions(init_pos)

    def _should_exit_early(self, time_start, time_max):
        """Check if simulation should exit early due to time limit."""
        return time.time() - time_start > time_max

    def _run_trajectory_iteration(
        self,
        iteration,
        save_freq,
        tag,
        h5_file,
        h5_freq,
        precision,
        burn_in,
        string_count,
        h5_string,
    ):
        """Run a single trajectory iteration."""
        # Run simulation step
        self.run_sim(save_freq)

        # Get system information
        positions, velocities, forces, pe, ke, cell, cvs = self.get_information()

        # Save energy data
        self._save_energy_data(tag, pe)

        # Write trajectory frame
        h5_file.write_frame(positions, velocities, forces, pe, ke, cell, cvs)

        # Handle file rotation
        if self._should_rotate_files(iteration, h5_freq):
            h5_file = self._rotate_trajectory_files(h5_file, h5_freq, precision)

        # Handle string method updates
        if self.string_freq is not None:
            string_count = self._handle_string_method_update(
                iteration, burn_in, string_count, h5_string
            )

        return string_count, h5_file

    def _save_energy_data(self, tag, pe):
        """Save potential energy data."""
        with open(f"{tag}_pe.txt", "ab") as f:
            np.savetxt(f, np.expand_dims(pe, 0), fmt="%.9e")

    def _should_rotate_files(self, iteration, h5_freq):
        """Check if trajectory files should be rotated."""
        return iteration % h5_freq == 0 and iteration != 0

    def _rotate_trajectory_files(self, h5_file, h5_freq, precision):
        """Rotate trajectory files."""
        self.simulation.reporters[1].close()
        self.internal_count += 1

        # Create new HDF5 reporter
        self.simulation.reporters[1] = HDF5Reporter(
            f"{self.base_filename_2}_{self.internal_count}.h5",
            self.save_int,
        )

        # Close old file and create new one
        h5_file.close()
        return TrajWriter(
            f"{self.base_filename_2}_{self.internal_count}_data.h5",
            self.num_atoms,
            h5_freq,
            precision=precision,
            cvs=self.string_forces,
        )

    def _cleanup_trajectory_files(self, h5_file, h5_string):
        """Clean up trajectory files."""
        self.simulation.reporters[1].close()
        h5_file.early_close()
        h5_string.close()

    def _should_update_string(self, iteration, burn_in):
        """Check if string should be updated at this iteration."""
        return (
            iteration % self.string_freq == 0
            and iteration != 0
            and iteration / self.string_freq > burn_in
        )

    def _should_reset_storage(self, iteration, burn_in):
        """Check if storage should be reset during burn-in."""
        return (
            iteration % self.string_freq == 0
            and iteration / self.string_freq <= burn_in
        )

    def _collect_current_cvs_and_forces(self):
        """Collect current CV values and forces from simulation context."""
        cvs0 = []
        for i in range(len(self.parameter_name)):
            cvs0.append(self.simulation.context.getParameter(self.parameter_name[i]))
        cvs0 = np.array(cvs0)

        forces0 = []
        for i in range(len(self.parameter_force_name)):
            forces0.append(
                self.simulation.context.getParameter(self.parameter_force_name[i])
            )
        forces0 = np.array(forces0)

        return cvs0, forces0

    def _process_cv_statistics(self, cvs0, forces0):
        """Process CV statistics with periodic boundary conditions."""
        cvs_store_np = np.array(self.cvs_store)
        cvs_store_np -= cvs0
        cvs_store_np = cvs_store_np - 2 * np.pi * np.rint(cvs_store_np / (2 * np.pi))

        mean_cvs_x = np.mean(np.cos(cvs_store_np), axis=0)
        mean_cvs_y = np.mean(np.sin(cvs_store_np), axis=0)
        cvs_diff = np.arctan2(mean_cvs_y, mean_cvs_x)
        cvs_diff *= forces0

        metric = np.mean(self.metric_store, axis=0)
        inv_metric = np.linalg.inv(metric)

        return cvs_diff, metric, inv_metric

    def _calculate_gradient_with_projection(self, cvs0, cvs_diff, metric):
        """Calculate gradient with projection for string method."""
        cvs0_global = np.zeros((self.size, len(cvs0)))
        self.comm.Allgather(cvs0, cvs0_global)

        if self.rank == 0 or self.rank == self.size - 1:
            metric = np.ones_like(metric)

        grad = -metric @ cvs_diff
        projection = np.eye(len(cvs_diff))

        # Calculate neighboring differences
        pos_diff = None
        pos_diff_grad = None
        neg_diff = None

        if self.rank < self.size - 1:
            pos_diff = cvs0_global[self.rank + 1] - cvs0
            pos_diff -= 2 * np.pi * np.rint(pos_diff / (2 * np.pi))
            pos_diff_size = np.linalg.norm(pos_diff)
            pos_diff_grad = np.dot(pos_diff, grad)

        if self.rank > 0:
            neg_diff = cvs0_global[self.rank - 1] - cvs0
            neg_diff -= 2 * np.pi * np.rint(neg_diff / (2 * np.pi))
            neg_diff_size = np.linalg.norm(neg_diff)

        # Apply projection based on gradient direction
        if pos_diff is not None and neg_diff is not None:
            if pos_diff_grad >= 0:
                projection -= np.outer(pos_diff, pos_diff) / pos_diff_size**2
            else:
                projection -= np.outer(neg_diff, neg_diff) / neg_diff_size**2

        grad = projection @ grad

        if not self.update_ends:
            if self.rank == 0 or self.rank == self.size - 1:
                grad = np.zeros_like(grad)

        return self.adam.update(grad), cvs0_global

    def _communicate_and_interpolate_string(
        self, cvs0, grad, inv_metric, string_count, h5_string
    ):
        """Handle MPI communication and string interpolation."""
        if self.comm is None:
            return cvs0

        cv_global = np.zeros((self.size, len(cvs0)))
        inv_metric_global = np.zeros((self.size, len(cvs0), len(cvs0)))
        grad_global = np.zeros((self.size, len(cvs0)))

        self.comm.Allgather(cvs0, cv_global)
        self.comm.Allgather(grad, grad_global)
        self.comm.Allgather(inv_metric, inv_metric_global)

        if self.rank == 0:
            cv_global = self._interpolate_string_on_root(
                cv_global, grad_global, inv_metric_global, string_count, h5_string
            )
            self.comm.Scatter(cv_global, cvs0, root=0)
        else:
            self.comm.Scatter(cv_global, cvs0, root=0)

        return cvs0

    def _interpolate_string_on_root(
        self, cv_global, grad_global, inv_metric_global, string_count, h5_string
    ):
        """Perform string interpolation on root process."""
        cv_global_pre = cv_global.copy()
        t = np.linspace(0, 1, self.size)

        # Handle periodic boundary conditions
        for i in range(1, self.size):
            for j in range(len(cv_global[0])):
                if cv_global[i, j] - cv_global[i - 1, j] > np.pi:
                    cv_global[i:, j] = cv_global[i:, j] - 2 * np.pi
                elif cv_global[i, j] - cv_global[i - 1, j] < -np.pi:
                    cv_global[i:, j] = cv_global[i:, j] + 2 * np.pi

        # Thomas algorithm
        cv_global = thomas_inverse_batch_d(
            self.a_vec,
            self.b_vec,
            self.c_vec,
            cv_global - grad_global,
        )

        # Equal spacing in CV-space
        cv_global = self._apply_equal_spacing(cv_global, inv_metric_global, t)

        # Convert back to [-pi, pi] range
        cv_global = cv_global - 2 * np.pi * np.rint(cv_global / (2 * np.pi))

        self._save_string_data(
            cv_global,
            cv_global_pre,
            inv_metric_global,
            grad_global,
            string_count,
            h5_string,
        )

        return cv_global

    def _apply_equal_spacing(self, cv_global, inv_metric_global, t):
        """Apply equal spacing to the string in CV-space."""
        arc_length = np.zeros(self.size)
        delta_cv = cv_global[1:] - cv_global[:-1]

        if self.cv_weights is None:
            segment_lengths = np.sqrt(
                np.einsum(
                    "ni,nij,nj->n",
                    delta_cv,
                    inv_metric_global[:-1],
                    delta_cv,
                )
            )
        else:
            delta_cv *= self.cv_weights[np.newaxis, :]
            segment_lengths = np.linalg.norm(delta_cv, axis=1)

        arc_length[1:] = np.cumsum(segment_lengths)
        arc_length /= arc_length[-1]
        equal_spaced_arc_lengths = np.linspace(0, 1, self.size)
        t_equal = np.interp(equal_spaced_arc_lengths, arc_length, t)

        return interp1d(t, cv_global, axis=0, kind="linear")(t_equal)

    def _save_string_data(
        self,
        cv_global,
        cv_global_pre,
        inv_metric_global,
        grad_global,
        string_count,
        h5_string,
    ):
        """Save string data to HDF5 file."""
        h5_string.create_group(f"config_{string_count}")
        h5_string[f"config_{string_count}"].create_dataset(
            "cvs", data=cv_global, dtype=np.float64
        )
        h5_string[f"config_{string_count}"].create_dataset(
            "cvs_pre", data=cv_global_pre, dtype=np.float64
        )
        h5_string[f"config_{string_count}"].create_dataset(
            "inv_metric", data=inv_metric_global, dtype=np.float64
        )
        h5_string[f"config_{string_count}"].create_dataset(
            "internal_count", data=self.internal_count, dtype=int
        )
        h5_string[f"config_{string_count}"].create_dataset(
            "grad", data=grad_global, dtype=np.float64
        )

    def _update_simulation_parameters(self, cvs0):
        """Update simulation parameters with new CV values."""
        for i in range(len(self.parameter_name)):
            self.simulation.context.setParameter(self.parameter_name[i], cvs0[i])

    def _handle_string_method_update(self, iteration, burn_in, string_count, h5_string):
        """Handle string method updates during trajectory generation."""
        if self._should_update_string(iteration, burn_in):
            # Collect current state
            cvs0, forces0 = self._collect_current_cvs_and_forces()

            # Process statistics
            cvs_diff, metric, inv_metric = self._process_cv_statistics(cvs0, forces0)

            # Calculate gradient
            grad, _ = self._calculate_gradient_with_projection(cvs0, cvs_diff, metric)

            # Communicate and interpolate
            cvs0 = self._communicate_and_interpolate_string(
                cvs0, grad, inv_metric, string_count, h5_string
            )

            # Update parameters
            self._update_simulation_parameters(cvs0)

            # Reset storage
            self.cvs_store = []
            self.metric_store = []

            return string_count + 1

        elif self._should_reset_storage(iteration, burn_in):
            self.cvs_store = []
            self.metric_store = []

        return string_count


class Adam:
    """Adam optimizer for updating parameters."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, grads):
        """Returns gradient for parameter optimization using Adam."""
        if self.m is None:
            self.m = np.zeros_like(grads)
            self.v = np.zeros_like(grads)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        grad_adam = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return grad_adam


@numba.njit(cache=True, parallel=True)
def thomas_inverse_batch_d(a, b, c, d):
    """
    Solves a batch of tridiagonal systems Ax=d, where A is the same for
    all systems but d varies.

    Arguments:
        a: A 1D numpy array of shape (n-1,) corresponding to the sub-diagonal of the tridiagonal matrix
        b: A 1D numpy array of shape (n,) corresponding to the diagonal of the tridiagonal matrix
        c: A 1D numpy array of shape (n-1,) corresponding to the super-diagonal of the tridiagonal matrix
        d: A 2D numpy array of shape (n, b) where n is the number of equations and b is the batch size
    """  # noqa: D205
    n = b.shape[0]
    bs = d.shape[1]  # Batch size

    c_prime = np.zeros(n - 1, dtype=np.float64)
    d_prime = np.zeros((n, bs), dtype=np.float64)
    x = np.zeros((n, bs), dtype=np.float64)

    # --- Forward elimination ---
    b0_inv = 1.0 / b[0]
    c_prime[0] = c[0] * b0_inv
    d_prime[0, :] = d[0, :] * b0_inv

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        denom_inv = 1.0 / denom
        c_prime[i] = c[i] * denom_inv
        d_prime[i, :] = (d[i, :] - a[i - 1] * d_prime[i - 1, :]) * denom_inv

    denom_last = b[n - 1] - a[n - 2] * c_prime[n - 2]
    denom_last_inv = 1.0 / denom_last
    d_prime[n - 1, :] = (d[n - 1, :] - a[n - 2] * d_prime[n - 2, :]) * denom_last_inv

    # --- Backward substitution (Batched) ---
    x[n - 1, :] = d_prime[n - 1, :]
    for i in range(n - 2, -1, -1):
        x[i, :] = d_prime[i, :] - c_prime[i] * x[i + 1, :]

    return x
