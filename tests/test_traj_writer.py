"""Unit tests for TrajWriter class."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from omm_fts.io.traj_writer import PRECISION_32, PRECISION_64, TrajWriter


class TestTrajWriter:
    """Test suite for TrajWriter class."""

    def test_init_creates_file(self):
        """Test that initialization creates an HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            writer = TrajWriter(filename, num_atoms=10, num_frames=100)

            assert Path(filename).exists()
            writer.close()

    def test_init_creates_datasets(self):
        """Test that initialization creates required datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            writer = TrajWriter(filename, num_atoms=10, num_frames=100)

            expected_datasets = [
                "positions",
                "velocities",
                "forces",
                "pe",
                "ke",
                "cell",
            ]
            for dataset in expected_datasets:
                assert dataset in writer.file

            writer.close()

    def test_init_with_cvs(self):
        """Test initialization with collective variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            cvs = [None, None, None]  # 3 CVs
            writer = TrajWriter(filename, num_atoms=10, num_frames=100, cvs=cvs)

            assert "cv_0" in writer.file
            assert "cv_1" in writer.file
            assert "cv_2" in writer.file

            writer.close()

    def test_init_with_rank(self):
        """Test initialization with rank tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            writer = TrajWriter(filename, num_atoms=10, num_frames=100, rank=True)

            assert "rank" in writer.file

            writer.close()

    def test_precision_32(self):
        """Test 32-bit precision datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            writer = TrajWriter(
                filename, num_atoms=10, num_frames=100, precision=PRECISION_32
            )

            assert writer.file["positions"].dtype == np.float32
            assert writer.file["pe"].dtype == np.float32

            writer.close()

    def test_precision_64(self):
        """Test 64-bit precision datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            writer = TrajWriter(
                filename, num_atoms=10, num_frames=100, precision=PRECISION_64
            )

            assert writer.file["positions"].dtype == np.float64
            assert writer.file["pe"].dtype == np.float64

            writer.close()

    def test_invalid_precision_raises_error(self):
        """Test that invalid precision raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            with pytest.raises(ValueError, match="Precision must be 32 or 64"):
                TrajWriter(filename, num_atoms=10, num_frames=100, precision=16)

    def test_write_frame(self):
        """Test writing a single frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            num_atoms = 10
            writer = TrajWriter(
                filename, num_atoms=num_atoms, num_frames=100, precision=64
            )

            positions = np.random.rand(num_atoms, 3)
            velocities = np.random.rand(num_atoms, 3)
            forces = np.random.rand(num_atoms, 3)
            pe = 1.5
            ke = 2.3
            cell = np.eye(3)

            writer.write_frame(positions, velocities, forces, pe, ke, cell)

            # Use allclose instead of array_equal for floating point comparisons
            np.testing.assert_allclose(
                writer.file["positions"][0], positions, rtol=1e-7
            )
            np.testing.assert_allclose(
                writer.file["velocities"][0], velocities, rtol=1e-7
            )
            np.testing.assert_allclose(writer.file["forces"][0], forces, rtol=1e-7)
            assert writer.file["pe"][0, 0] == pe
            assert writer.file["ke"][0, 0] == ke
            np.testing.assert_array_equal(writer.file["cell"][0], cell)
            assert writer.frame == 1

            writer.close()

    def test_write_multiple_frames(self):
        """Test writing multiple frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            num_atoms = 5
            num_frames = 10
            writer = TrajWriter(filename, num_atoms=num_atoms, num_frames=num_frames)

            for i in range(num_frames):
                positions = np.ones((num_atoms, 3)) * i
                velocities = np.ones((num_atoms, 3)) * i
                forces = np.ones((num_atoms, 3)) * i
                pe = float(i)
                ke = float(i)
                cell = np.eye(3) * i

                writer.write_frame(positions, velocities, forces, pe, ke, cell)

            assert writer.frame == num_frames
            writer.close()

    def test_write_frame_with_cvs(self):
        """Test writing frame with collective variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            num_atoms = 10
            cvs = [None, None]
            writer = TrajWriter(filename, num_atoms=num_atoms, num_frames=100, cvs=cvs)

            positions = np.random.rand(num_atoms, 3)
            velocities = np.random.rand(num_atoms, 3)
            forces = np.random.rand(num_atoms, 3)
            pe = 1.5
            ke = 2.3
            cell = np.eye(3)
            cvs_values = [0.5, 0.7]

            writer.write_frame(
                positions, velocities, forces, pe, ke, cell, cvs=cvs_values
            )

            assert writer.file["cv_0"][0, 0] == cvs_values[0]
            assert writer.file["cv_1"][0, 0] == cvs_values[1]

            writer.close()

    def test_write_frame_with_rank(self):
        """Test writing frame with rank information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            num_atoms = 10
            writer = TrajWriter(
                filename, num_atoms=num_atoms, num_frames=100, rank=True
            )

            positions = np.random.rand(num_atoms, 3)
            velocities = np.random.rand(num_atoms, 3)
            forces = np.random.rand(num_atoms, 3)
            pe = 1.5
            ke = 2.3
            cell = np.eye(3)
            rank = 5

            writer.write_frame(positions, velocities, forces, pe, ke, cell, rank=rank)

            assert writer.file["rank"][0, 0] == rank

            writer.close()

    def test_close(self):
        """Test closing the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            writer = TrajWriter(filename, num_atoms=10, num_frames=100)
            writer.close()

            # Verify file was written and is accessible
            assert Path(filename).exists()

            # Verify we can reopen the file (it was properly closed)
            with h5py.File(filename, "r") as f:
                assert "positions" in f

    def test_early_close_resizes_datasets(self):
        """Test early close resizes datasets to actual frame count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            num_atoms = 10
            num_frames = 100
            actual_frames = 5

            writer = TrajWriter(filename, num_atoms=num_atoms, num_frames=num_frames)

            for _ in range(actual_frames):
                positions = np.random.rand(num_atoms, 3)
                velocities = np.random.rand(num_atoms, 3)
                forces = np.random.rand(num_atoms, 3)
                pe = 1.5
                ke = 2.3
                cell = np.eye(3)
                writer.write_frame(positions, velocities, forces, pe, ke, cell)

            writer.early_close()

            # Verify resized datasets
            with h5py.File(filename, "r") as f:
                assert f["positions"].shape[0] == actual_frames
                assert f["velocities"].shape[0] == actual_frames
                assert f["forces"].shape[0] == actual_frames
                assert f["pe"].shape[0] == actual_frames
                assert f["ke"].shape[0] == actual_frames
                assert f["cell"].shape[0] == actual_frames

    def test_early_close_with_cvs(self):
        """Test early close with CVs resizes CV datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            num_atoms = 10
            num_frames = 100
            actual_frames = 3
            cvs = [None, None]

            writer = TrajWriter(
                filename, num_atoms=num_atoms, num_frames=num_frames, cvs=cvs
            )

            for _ in range(actual_frames):
                positions = np.random.rand(num_atoms, 3)
                velocities = np.random.rand(num_atoms, 3)
                forces = np.random.rand(num_atoms, 3)
                pe = 1.5
                ke = 2.3
                cell = np.eye(3)
                cvs_values = [0.5, 0.7]
                writer.write_frame(
                    positions, velocities, forces, pe, ke, cell, cvs=cvs_values
                )

            writer.early_close()

            with h5py.File(filename, "r") as f:
                assert f["cv_0"].shape[0] == actual_frames
                assert f["cv_1"].shape[0] == actual_frames

    def test_early_close_with_rank(self):
        """Test early close with rank resizes rank dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = str(Path(tmpdir) / "test.h5")
            num_atoms = 10
            num_frames = 100
            actual_frames = 3

            writer = TrajWriter(
                filename, num_atoms=num_atoms, num_frames=num_frames, rank=True
            )

            for i in range(actual_frames):
                positions = np.random.rand(num_atoms, 3)
                velocities = np.random.rand(num_atoms, 3)
                forces = np.random.rand(num_atoms, 3)
                pe = 1.5
                ke = 2.3
                cell = np.eye(3)
                writer.write_frame(positions, velocities, forces, pe, ke, cell, rank=i)

            writer.early_close()

            with h5py.File(filename, "r") as f:
                assert f["rank"].shape[0] == actual_frames
