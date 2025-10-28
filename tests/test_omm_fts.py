"""Unit tests for OMMFF class."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from omm_fts.omm.omm_fts import OMMFF, Adam, thomas_inverse_batch_d


class TestAdam:
    """Test suite for Adam optimizer."""

    def test_initialization(self):
        """Test Adam optimizer initialization."""
        adam = Adam(lr=0.001, beta1=0.9, beta2=0.999)
        assert adam.lr == 0.001  # noqa: PLR2004
        assert adam.beta1 == 0.9  # noqa: PLR2004
        assert adam.beta2 == 0.999  # noqa: PLR2004
        assert adam.m is None
        assert adam.v is None
        assert adam.t == 0

    def test_first_update(self):
        """Test first gradient update initializes m and v."""
        adam = Adam(lr=0.01)
        grads = np.array([1.0, 2.0, 3.0])

        result = adam.update(grads)

        assert adam.m is not None
        assert adam.v is not None
        assert adam.t == 1
        assert result.shape == grads.shape

    def test_multiple_updates(self):
        """Test multiple gradient updates."""
        adam = Adam(lr=0.01)
        grads1 = np.array([1.0, 2.0])
        grads2 = np.array([0.5, 1.5])

        result1 = adam.update(grads1)
        result2 = adam.update(grads2)

        assert adam.t == 2  # noqa: PLR2004
        assert result1.shape == grads1.shape
        assert result2.shape == grads2.shape

    def test_update_with_zero_gradients(self):
        """Test update with zero gradients."""
        adam = Adam(lr=0.01)
        grads = np.zeros(5)

        result = adam.update(grads)

        np.testing.assert_array_almost_equal(result, np.zeros(5))


class TestThomasInverse:
    """Test suite for thomas_inverse_batch_d function."""

    def test_simple_tridiagonal_system(self):
        """Test solving a simple tridiagonal system."""
        n = 5
        a = -np.ones(n - 1)
        b = 2 * np.ones(n)
        c = -np.ones(n - 1)
        d = np.ones((n, 1))

        result = thomas_inverse_batch_d(a, b, c, d)

        assert result.shape == (n, 1)

    def test_batch_systems(self):
        """Test solving multiple tridiagonal systems."""
        n = 4
        batch_size = 3
        a = -np.ones(n - 1)
        b = 2 * np.ones(n)
        c = -np.ones(n - 1)
        d = np.random.rand(n, batch_size)

        result = thomas_inverse_batch_d(a, b, c, d)

        assert result.shape == (n, batch_size)

    def test_identity_system(self):
        """Test system that is essentially identity."""
        n = 3
        a = np.zeros(n - 1)
        b = np.ones(n)
        c = np.zeros(n - 1)
        d = np.array([[1.0], [2.0], [3.0]])

        result = thomas_inverse_batch_d(a, b, c, d)

        np.testing.assert_array_almost_equal(result, d)


class TestOMMFF:
    """Test suite for OMMFF class."""

    @pytest.fixture
    def mock_system(self):
        """Create a mock OpenMM system."""
        system = Mock()
        system.system = Mock()
        system.system.getNumParticles = Mock(return_value=2)
        system.positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        system.topology = Mock()

        # Mock particle masses
        system.system.getParticleMass = Mock(
            side_effect=lambda i: Mock(value_in_unit_system=lambda x: 1.0)
        )

        return system

    @patch("omm_fts.omm.omm_fts.HDF5Reporter")
    @patch("omm_fts.omm.omm_fts.CheckpointReporter")
    def test_initialization_new_simulation(
        self, mock_checkpoint, mock_reporter, mock_system, tmp_path
    ):
        """Test OMMFF initialization for a new simulation."""
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

            # Use temp directory for file operations
            folder = str(tmp_path / "test")

            # Create OMMFF instance with langevin integrator to avoid CSVR complexities
            ommff = OMMFF(
                system=mock_system,
                platform="CPU",
                folder_name=folder,
                save_int=10,
                comm=None,
                integrator_name="langevin",  # Use simpler integrator for testing
                minimize_init=False,  # Skip complex minimization logic
            )

            assert ommff.count == 0
            assert ommff.num_atoms == 2  # noqa: PLR2004
            assert ommff.comm is None
            assert mock_simulation.minimizeEnergy.called

    @patch("omm_fts.omm.omm_fts.HDF5Reporter")
    @patch("omm_fts.omm.omm_fts.CheckpointReporter")
    def test_run_sim(self, mock_checkpoint, mock_reporter, mock_system, tmp_path):
        """Test running simulation steps."""
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
            mock_simulation.step = Mock()

            folder = str(tmp_path / "test")
            ommff = OMMFF(
                system=mock_system,
                folder_name=folder,
                save_int=0,
                comm=None,
                integrator_name="langevin",
                minimize_init=False,
            )
            ommff.run_sim(steps=100)

            mock_simulation.step.assert_called_once_with(100)

    @patch("omm_fts.omm.omm_fts.HDF5Reporter")
    @patch("omm_fts.omm.omm_fts.CheckpointReporter")
    def test_initialization_with_mpi(
        self, mock_checkpoint, mock_reporter, mock_system, tmp_path
    ):
        """Test OMMFF initialization with MPI communicator."""
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

            # Mock MPI communicator
            mock_comm = Mock()
            mock_comm.Get_rank.return_value = 0
            mock_comm.Get_size.return_value = 4

            folder = str(tmp_path / "test")
            ommff = OMMFF(
                system=mock_system,
                platform="CPU",
                folder_name=folder,
                save_int=10,
                comm=mock_comm,
                integrator_name="langevin",
                minimize_init=False,
            )

            assert ommff.rank == 0
            assert ommff.size == 4  # noqa: PLR2004
            assert ommff.a_vec.shape == (3,)  # size - 1
            assert ommff.b_vec.shape == (4,)  # size
            assert ommff.c_vec.shape == (3,)  # size - 1

    def test_setup_filenames_and_count_new(self, tmp_path):
        """Test filename setup for new simulation."""
        folder = str(tmp_path / "sim")
        ommff = OMMFF.__new__(OMMFF)
        ommff._setup_filenames_and_count(folder)

        assert ommff.count == 0
        assert ommff.folder_name == folder

    def test_setup_filenames_and_count_existing(self, tmp_path):
        """Test filename setup with existing count file."""
        folder = str(tmp_path / "sim")
        count_file = tmp_path / "sim_count.txt"
        count_file.write_text("5")

        ommff = OMMFF.__new__(OMMFF)
        ommff._setup_filenames_and_count(folder)

        assert ommff.count == 6  # noqa : PLR2004
