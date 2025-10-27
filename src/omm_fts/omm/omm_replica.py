"""OpenMM interface for doing Hamiltonian replica exchange simulations."""

import time
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numba  # type: ignore[import-untyped]
import numpy as np
from mdtraj.reporters import HDF5Reporter  # type: ignore[import-untyped]
from numpy.typing import NDArray
from openmm.unit import (  # type: ignore[import-untyped]
    AVOGADRO_CONSTANT_NA,
    BOLTZMANN_CONSTANT_kB,
    kelvin,
    md_unit_system,
    picoseconds,
)

from omm_fts.io.traj_writer import TrajWriter
from omm_fts.omm.omm_fts import OMMFF


class OMMFFReplica(OMMFF):
    """OMM Interface for Hamiltonian replica exchange simulations."""

    def __init__(
        self,
        system: Any,
        platform: str = "CUDA",
        precision: str = "single",
        integrator_name: str = "csvr_leapfrog",
        temperature: Any = 300.0 * kelvin,
        velocities_com: bool = True,
        time_step: Any = 0.001 * picoseconds,
        friction: Any = 0.1 / picoseconds,
        seed: int = 1,
        save_int: int = 10,
        folder_name: str = "",
        custom_forces: Any = None,
        string_forces: Any = None,
        force_groups: list[int] | None = None,
        parameter_name: list[str] | None = None,
        parameter_force_name: list[str] | None = None,
        comm: Any = None,
        swap_scheme: str = "mixing",
    ) -> None:
        # Initialize parent class
        super().__init__(
            system=system,
            platform=platform,
            precision=precision,
            integrator_name=integrator_name,
            temperature=temperature,
            velocities_com=velocities_com,
            time_step=time_step,
            friction=friction,
            seed=seed,
            save_int=save_int,
            folder_name=folder_name,
            custom_forces=custom_forces,
            string_forces=string_forces,
            force_groups=force_groups,
            parameter_name=parameter_name,
            parameter_force_name=parameter_force_name,
            string_freq=None,  # Not used in replica exchange
            string_dt=None,  # Not used in replica exchange
            string_kappa=None,  # Not used in replica exchange
            cv_weights=None,  # Not used in replica exchange
            update_ends=True,  # Not used in replica exchange
            comm=comm,
            minimize_init=False,  # Not used in replica exchange
            minimize_intervals=None,  # Not used in replica exchange
        )

        # Initialize replica-specific attributes
        self._setup_swap_scheme(swap_scheme, temperature)

    def _setup_swap_scheme(self, swap_scheme: str, temperature: Any) -> None:
        """Sets up the swap scheme for the simulation."""
        if swap_scheme not in ["mixing", "neighbors"]:
            raise ValueError(
                f"swap_scheme must be 'mixing' or 'neighbors', got '{swap_scheme}'"
            )
        self.swap_scheme = swap_scheme
        self.beta = 1.0 / (temperature * BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA)
        self.beta = self.beta.value_in_unit_system(md_unit_system)
        self.num_attempted: NDArray[Any] | None = None
        self.num_accepted: NDArray[Any] | None = None

        if self.comm is not None:
            self.nswap_attemps = self.size**3
            assert self.parameter_name is not None, "parameter_name must be set"
            assert self.parameter_force_name is not None, (
                "parameter_force_name must be set"
            )
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

            # Initialize replica rank
            self.replica_rank = self.rank

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

    def get_information(
        self, as_numpy: bool = True, enforce_periodic_box: bool = True
    ) -> tuple[
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
        float,
        float,
        NDArray[Any],
        list[float],
    ]:
        """Gets information (positions, forces and PE of system).

        Overrides parent method to handle replica-specific CV collection.
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

        cvs_list: list[float] = []
        if self.string_forces is not None:
            # turn on track
            self.simulation.context.setParameter("track", 1)
            assert self.force_groups is not None, "force_groups must be set"
            for i in range(len(self.string_forces)):
                state = self.simulation.context.getState(
                    getEnergy=True,
                    groups={self.force_groups[i]},
                )
                cvs_list.append(
                    state.getPotentialEnergy().value_in_unit_system(md_unit_system)
                )

            self.cvs_store.append(cvs_list)
            # turn off track
            self.simulation.context.setParameter("track", 0)

        return positions, velocities, forces, pe, ke, cell, cvs_list

    def generate_long_trajectory(
        self,
        init_pos: NDArray[Any] | None = None,
        num_data_points: int = int(1e8),
        save_freq: int = 1000,
        h5_freq: int = 100,
        enforce_periodic_box: bool = True,
        tag: str | None = None,
        time_max: float = 117 * 60,
        precision: int = 32,
        burn_in: int = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        swap_freq: int = 1,
    ) -> None:
        """Generates long trajectory with replica exchange swaps."""
        # Setup phase
        tag = self._setup_trajectory_tag(tag)
        start_iter = self._setup_trajectory_start(tag)
        time_start = time.time()

        # Initialize trajectory files (replica-specific)
        h5_file = self._setup_replica_trajectory_files(start_iter, h5_freq, precision)

        # Set initial positions if provided
        self._set_initial_positions(init_pos)

        # Main simulation loop
        for iteration in range(start_iter, num_data_points):
            if self._should_exit_early(time_start, time_max):
                self._cleanup_replica_trajectory_files(h5_file)
                self.comm.Barrier()
                return

            h5_file = self._run_replica_trajectory_iteration(
                iteration, save_freq, tag, h5_file, h5_freq, precision, swap_freq
            )

        # Natural completion cleanup
        self._cleanup_replica_trajectory_files(h5_file)

    def _setup_replica_trajectory_files(
        self, start_iter: int, h5_freq: int, precision: int
    ) -> TrajWriter:
        """Setup replica-specific trajectory files."""
        h5_chunk = -(start_iter % h5_freq) + h5_freq + 1
        return TrajWriter(
            f"{self.base_filename_2}_{self.internal_count}_data.h5",
            self.num_atoms,
            h5_chunk,
            precision=precision,
            cvs=self.string_forces,
            rank=True,
        )

    def _run_replica_trajectory_iteration(
        self,
        iteration: int,
        save_freq: int,
        tag: str,
        h5_file: TrajWriter,
        h5_freq: int,
        precision: int,
        swap_freq: int,
    ) -> TrajWriter:
        """Run a single replica trajectory iteration."""
        # Run simulation step
        self.run_sim(save_freq)

        # Get system information
        positions, velocities, forces, pe, ke, cell, cvs = self.get_information()

        # Save energy data
        self._save_energy_data(tag, pe)

        # Handle replica swaps
        if self._should_attempt_swap(iteration, swap_freq):
            # Convert cvs list to numpy array for swap operation
            cvs_array = np.array(cvs, dtype=np.float64)
            self._perform_replica_swap(cvs_array)

        # Write trajectory frame with replica rank
        h5_file.write_frame(
            positions, velocities, forces, pe, ke, cell, cvs=cvs, rank=self.replica_rank
        )

        # Handle file rotation
        if self._should_rotate_files(iteration, h5_freq):
            h5_file = self._rotate_replica_trajectory_files(h5_file, h5_freq, precision)

        return h5_file

    def _should_attempt_swap(self, iteration: int, swap_freq: int) -> bool:
        """Check if replica swap should be attempted."""
        return iteration % swap_freq == 0 and self.comm is not None and iteration != 0

    def _rotate_replica_trajectory_files(
        self, h5_file: TrajWriter, h5_freq: int, precision: int
    ) -> TrajWriter:
        """Rotate replica trajectory files."""
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
            rank=True,
        )

    def _cleanup_replica_trajectory_files(self, h5_file: TrajWriter) -> None:
        """Clean up replica trajectory files."""
        self.simulation.reporters[1].close()
        h5_file.early_close()

    def _gather_replica_data(
        self, cvs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[Any]]:
        """Gather collective variables and replica ranks from all processes."""
        cvs_all = np.zeros((self.size, len(cvs)), dtype=np.float64)
        self.comm.Allgather(cvs, cvs_all)

        replica_rank_all = np.zeros(self.size, dtype=np.int64)
        self.comm.Allgather(
            np.array(self.replica_rank, dtype=np.int64), replica_rank_all
        )

        return cvs_all, replica_rank_all

    def _update_replica_parameters(self, replica_rank_new: NDArray[Any]) -> None:
        """Update simulation context parameters based on new replica assignment."""
        assert self.parameter_name is not None, "parameter_name must be set"
        assert self.parameter_force_name is not None, "parameter_force_name must be set"
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

    def _save_swap_results(self, replica_rank_new: NDArray[Any]) -> None:
        """Save replica ranks and swap rates to HDF5 files."""
        with h5py.File("replica_ranks.h5", "a") as f:
            dset_name = f"config_{self.count}_{len(f.keys())}"
            f.create_dataset(dset_name, data=replica_rank_new)

        with h5py.File("swap_rates.h5", "a") as f:
            group = f.create_group(f"config_{self.count}_{len(f.keys())}")
            group.create_dataset("num_accepted", data=self.num_accepted)
            group.create_dataset("num_attempted", data=self.num_attempted)

    def _perform_replica_swap(self, cvs: NDArray[np.float64]) -> None:
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


@numba.njit  # type: ignore[misc]
def mix_replicas(
    beta: float,
    cvs_all: NDArray[np.float64],
    parameter_values: NDArray[np.float64],
    force_values: NDArray[np.float64],
    replica_rank: NDArray[Any],
    nswap_attemps: int,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Mixes replicas based on the Metropolis criterion.

    Arguments:
        beta: A float corresponding to the inverse temperature
        cvs_all: A numpy array of shape (n_replicas, n_cvs) corresponding
            to the collective variables of all replicas
        parameter_values: A numpy array of shape (n_replicas, n_parameters)
            corresponding to the parameter values of all replicas
        force_values: A numpy array of shape (n_replicas, n_forces)
            corresponding to the force values of all replicas
        replica_rank: An int corresponding to the rank of the current replica
        nswap_attemps: An int specifying the number of swap attempts
    Returns:
        replica_rank: An int corresponding to the new rank of the replica
            after mixing
        num_accepted: A numpy array of shape (n_replicas, n_replicas)
            corresponding to the number of accepted swaps between replicas
        num_attempted: A numpy array of shape (n_replicas, n_replicas)
            corresponding to the number of attempted swaps between replicas
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


@numba.njit  # type: ignore[misc]
def mix_neighboring_replicas(
    beta: float,
    cvs_all: NDArray[np.float64],
    parameter_values: NDArray[np.float64],
    force_values: NDArray[np.float64],
    replica_rank: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Mixes replicas based on the Metropolis criterion.

    Arguments:
        beta: A float corresponding to the inverse temperature
        cvs_all: A numpy array of shape (n_replicas, n_cvs) corresponding
            to the collective variables of all replicas
        parameter_values: A numpy array of shape (n_replicas, n_parameters)
            corresponding to the parameter values of all replicas
        force_values: A numpy array of shape (n_replicas, n_forces)
            corresponding to the force values of all replicas
        replica_rank: An int corresponding to the rank of the current replica
    Returns:
        replica_rank: An int corresponding to the new rank of the replica
            after mixing
        num_accepted: A numpy array of shape (n_replicas, n_replicas)
            corresponding to the number of accepted swaps between replicas
        num_attempted: A numpy array of shape (n_replicas, n_replicas)
            corresponding to the number of attempted swaps between replicas
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
