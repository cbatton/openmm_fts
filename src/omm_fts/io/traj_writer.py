"""A class to write trajectory data to a file in HDF5 format."""

from typing import Any

import h5py  # type: ignore[import-untyped]
from numpy.typing import NDArray

# Constants for precision
PRECISION_32 = 32
PRECISION_64 = 64


class TrajWriter:
    """Class to write trajectory data to a file."""

    def __init__(
        self,
        filename: str,
        num_atoms: int,
        num_frames: int,
        precision: int = 32,
        cvs: list[Any] | None = None,
        rank: bool | None = None,
    ) -> None:
        self.filename = filename
        self.num_atoms = num_atoms
        self.num_frames = num_frames
        self.precision = precision
        self.file: h5py.File = h5py.File(self.filename, "w")

        if self.precision == PRECISION_32:
            precision_str = "f4"
        elif self.precision == PRECISION_64:
            precision_str = "f8"
        else:
            raise ValueError("Precision must be 32 or 64")

        self.cvs = cvs
        self.rank = rank

        self.file.create_dataset(
            "positions",
            (self.num_frames, self.num_atoms, 3),
            dtype=precision_str,
            chunks=(1, self.num_atoms, 3),
        )
        self.file.create_dataset(
            "velocities",
            (self.num_frames, self.num_atoms, 3),
            dtype=precision_str,
            chunks=(1, self.num_atoms, 3),
        )
        self.file.create_dataset(
            "forces",
            (self.num_frames, self.num_atoms, 3),
            dtype=precision_str,
            chunks=(1, self.num_atoms, 3),
        )
        self.file.create_dataset(
            "pe", (self.num_frames, 1), dtype=precision_str, chunks=(1, 1)
        )
        self.file.create_dataset(
            "ke", (self.num_frames, 1), dtype=precision_str, chunks=(1, 1)
        )
        self.file.create_dataset(
            "cell", (self.num_frames, 3, 3), dtype=precision_str, chunks=(1, 3, 3)
        )
        if self.cvs is not None:
            for i in range(len(cvs)):  # type: ignore[arg-type]
                self.file.create_dataset(
                    f"cv_{i}", (self.num_frames, 1), dtype=precision_str, chunks=(1, 1)
                )
        if self.rank is not None:
            self.file.create_dataset(
                "rank", (self.num_frames, 1), dtype="i4", chunks=(1, 1)
            )
        self.frame = 0

    def write_frame(
        self,
        positions: NDArray[Any],
        velocities: NDArray[Any],
        forces: NDArray[Any],
        pe: float,
        ke: float,
        cell: NDArray[Any],
        cvs: list[float] | NDArray[Any] | None = None,
        rank: int | None = None,
    ) -> None:
        """Write a single frame of trajectory data to the file."""
        self.file["positions"][self.frame] = positions
        self.file["velocities"][self.frame] = velocities
        self.file["forces"][self.frame] = forces
        self.file["pe"][self.frame] = pe
        self.file["ke"][self.frame] = ke
        self.file["cell"][self.frame] = cell
        if cvs is not None:
            for i, cv in enumerate(cvs):
                self.file[f"cv_{i}"][self.frame] = cv
        if rank is not None:
            self.file["rank"][self.frame] = rank
        self.frame += 1

    def close(self) -> None:
        """Close the trajectory file."""
        self.file.close()

    def early_close(self) -> None:
        """Close the trajectory file early.

        Resizes datasets to the number of frames written.
        """
        self.file["positions"].resize(self.frame, axis=0)
        self.file["velocities"].resize(self.frame, axis=0)
        self.file["forces"].resize(self.frame, axis=0)
        self.file["pe"].resize(self.frame, axis=0)
        self.file["ke"].resize(self.frame, axis=0)
        self.file["cell"].resize(self.frame, axis=0)
        if self.cvs is not None:
            for i in range(len(self.cvs)):
                self.file[f"cv_{i}"].resize(self.frame, axis=0)
        if self.rank is not None:
            self.file["rank"].resize(self.frame, axis=0)
        self.file.close()
