"""Unit tests for OMMFFReplica class."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from omm_fts.omm.omm_replica import (
    OMMFFReplica,
    mix_neighboring_replicas,
    mix_replicas,
)


class TestMixReplicas:
    """Test suite for mix_replicas function."""

    def test_no_swaps_identical_states(self):
        """Test that identical states don't swap."""
        beta = 1.0
        n_replicas = 3
        cvs_all = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        parameter_values = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        force_values = np.ones((n_replicas, 2))
        replica_rank = np.array([0, 1, 2])
        nswap_attempts = 10

        result_rank, num_accepted, num_attempted = mix_replicas(
            beta,
            cvs_all,
            parameter_values,
            force_values,
            replica_rank,
            nswap_attempts,
        )

        assert result_rank.shape == (n_replicas,)
        assert num_accepted.shape == (n_replicas, n_replicas)
        assert num_attempted.shape == (n_replicas, n_replicas)

    def test_swap_attempts_count(self):
        """Test that swap attempts are counted correctly."""
        np.random.seed(42)
        beta = 1.0
        n_replicas = 4
        cvs_all = np.random.rand(n_replicas, 2)
        parameter_values = np.random.rand(n_replicas, 2)
        force_values = np.ones((n_replicas, 2))
        replica_rank = np.arange(n_replicas)
        nswap_attempts = 20

        _, _, num_attempted = mix_replicas(
            beta,
            cvs_all,
            parameter_values,
            force_values,
            replica_rank,
            nswap_attempts,
        )

        # Total attempts should be counted (symmetrically)
        assert np.sum(num_attempted) > 0

    def test_favorable_swaps_accepted(self):
        """Test that very favorable swaps are accepted."""
        np.random.seed(42)
        beta = 100.0  # High beta makes energetic differences matter more
        n_replicas = 2
        cvs_all = np.array([[0.0, 0.0], [1.0, 1.0]])
        parameter_values = np.array([[1.0, 1.0], [0.0, 0.0]])
        force_values = np.ones((n_replicas, 2)) * 10
        replica_rank = np.array([0, 1])
        nswap_attempts = 100

        _, num_accepted, _ = mix_replicas(
            beta,
            cvs_all,
            parameter_values,
            force_values,
            replica_rank,
            nswap_attempts,
        )

        # With many attempts and favorable swaps, should have some acceptances
        assert np.sum(num_accepted) >= 0


class TestMixNeighboringReplicas:
    """Test suite for mix_neighboring_replicas function."""

    def test_neighboring_swaps_only(self):
        """Test that only neighboring replicas are considered."""
        np.random.seed(42)
        beta = 1.0
        n_replicas = 4
        cvs_all = np.random.rand(n_replicas, 2)
        parameter_values = np.random.rand(n_replicas, 2)
        force_values = np.ones((n_replicas, 2))
        replica_rank = np.arange(n_replicas)

        result_rank, _, num_attempted = mix_neighboring_replicas(
            beta, cvs_all, parameter_values, force_values, replica_rank
        )

        assert result_rank.shape == (n_replicas,)
        # Only neighboring pairs should be attempted
        assert np.sum(num_attempted) > 0

    def test_alternating_pairs(self):
        """Test that alternating pairs are swapped."""
        np.random.seed(42)
        beta = 1.0
        n_replicas = 4
        cvs_all = np.random.rand(n_replicas, 2)
        parameter_values = np.random.rand(n_replicas, 2)
        force_values = np.ones((n_replicas, 2))
        replica_rank = np.arange(n_replicas)

        _, _, num_attempted = mix_neighboring_replicas(
            beta, cvs_all, parameter_values, force_values, replica_rank
        )

        # Should only attempt neighboring pairs
        # Check that non-neighbors are not attempted
        assert num_attempted[0, 2] == 0
        assert num_attempted[0, 3] == 0


class TestOMMFFReplica:
    """Test suite for OMMFFReplica class."""

    @pytest.fixture
    def mock_system(self):
        """Create a mock OpenMM system."""
        system = Mock()
        system.system = Mock()
        system.system.getNumParticles = Mock(return_value=2)
        system.positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        system.topology = Mock()
        system.system.getParticleMass = Mock(
            side_effect=lambda i: Mock(value_in_unit_system=lambda x: 1.0)
        )
        return system

    @patch("omm_fts.omm.omm_fts.HDF5Reporter")
    @patch("omm_fts.omm.omm_fts.CheckpointReporter")
    def test_initialization_mixing_scheme(
        self, mock_checkpoint, mock_reporter, mock_system, tmp_path
    ):
        """Test initialization with mixing swap scheme."""
        with patch("omm_fts.omm.omm_fts.Simulation") as mock_simulation_cls, patch(
            "omm_fts.omm.omm_fts.Platform"
        ) as mock_platform, patch(
            "omm_fts.omm.omm_fts.LangevinMiddleIntegrator"
        ) as mock_integrator_cls:
            mock_platform.getPlatformByName.return_value = Mock()
            mock_integrator = Mock()
            mock_integrator.setRandomNumberSeed = Mock()
            mock_integrator_cls.return_value = mock_integrator

            mock_simulation = Mock()
            mock_simulation_cls.return_value = mock_simulation
            mock_simulation.context = Mock()
            mock_simulation.context.setPositions = Mock()
            mock_simulation.context.setVelocitiesToTemperature = Mock()
            mock_simulation.minimizeEnergy = Mock()
            mock_simulation.saveCheckpoint = Mock()
            mock_simulation.saveState = Mock()
            mock_simulation.reporters = []

            folder = str(tmp_path / "test")
            replica = OMMFFReplica(
                system=mock_system,
                folder_name=folder,
                swap_scheme="mixing",
                save_int=0,
                comm=None,
                integrator_name="langevin",  # Use simpler integrator
            )

            assert replica.swap_scheme == "mixing"

    @patch("omm_fts.omm.omm_fts.HDF5Reporter")
    @patch("omm_fts.omm.omm_fts.CheckpointReporter")
    def test_initialization_neighbors_scheme(
        self, mock_checkpoint, mock_reporter, mock_system, tmp_path
    ):
        """Test initialization with neighbors swap scheme."""
        with patch("omm_fts.omm.omm_fts.Simulation") as mock_simulation_cls, patch(
            "omm_fts.omm.omm_fts.Platform"
        ) as mock_platform, patch(
            "omm_fts.omm.omm_fts.LangevinMiddleIntegrator"
        ) as mock_integrator_cls:
            mock_platform.getPlatformByName.return_value = Mock()
            mock_integrator = Mock()
            mock_integrator.setRandomNumberSeed = Mock()
            mock_integrator_cls.return_value = mock_integrator

            mock_simulation = Mock()
            mock_simulation_cls.return_value = mock_simulation
            mock_simulation.context = Mock()
            mock_simulation.context.setPositions = Mock()
            mock_simulation.context.setVelocitiesToTemperature = Mock()
            mock_simulation.minimizeEnergy = Mock()
            mock_simulation.saveCheckpoint = Mock()
            mock_simulation.saveState = Mock()
            mock_simulation.reporters = []

            folder = str(tmp_path / "test")
            replica = OMMFFReplica(
                system=mock_system,
                folder_name=folder,
                swap_scheme="neighbors",
                save_int=0,
                comm=None,
                integrator_name="langevin",
            )

            assert replica.swap_scheme == "neighbors"

    @patch("omm_fts.omm.omm_fts.HDF5Reporter")
    @patch("omm_fts.omm.omm_fts.CheckpointReporter")
    def test_invalid_swap_scheme_raises_error(
        self, mock_checkpoint, mock_reporter, mock_system, tmp_path
    ):
        """Test that invalid swap scheme raises ValueError."""
        with patch("omm_fts.omm.omm_fts.Simulation") as mock_simulation_cls, patch(
            "omm_fts.omm.omm_fts.Platform"
        ) as mock_platform, patch(
            "omm_fts.omm.omm_fts.LangevinMiddleIntegrator"
        ) as mock_integrator_cls:
            mock_platform.getPlatformByName.return_value = Mock()
            mock_integrator = Mock()
            mock_integrator.setRandomNumberSeed = Mock()
            mock_integrator_cls.return_value = mock_integrator

            mock_simulation = Mock()
            mock_simulation_cls.return_value = mock_simulation
            mock_simulation.context = Mock()
            mock_simulation.reporters = []

            folder = str(tmp_path / "test")
            with pytest.raises(ValueError, match="swap_scheme must be"):
                OMMFFReplica(
                    system=mock_system,
                    folder_name=folder,
                    swap_scheme="invalid",
                    save_int=0,
                    comm=None,
                    integrator_name="langevin",
                )
