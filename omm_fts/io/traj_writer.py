"""A class to write trajectory data to a file in HDF5 format."""

import h5py


class TrajWriter:
    """Class to write trajectory data to a file."""

    def __init__(
        self, filename, num_atoms, num_frames, precision=32, cvs=None, rank=None
    ):
        self.filename = filename
        self.num_atoms = num_atoms
        self.num_frames = num_frames
        self.precision = precision
        self.file = h5py.File(self.filename, "w")

        if self.precision == 32:
            precision_str = "f4"
        elif self.precision == 64:
            precision_str = "f8"
        else:
            raise ValueError("Precision must be 32 or 64")

        if cvs is not None:
            self.cvs = cvs
        else:
            self.cvs = None
        if rank is not None:
            self.rank = rank
        else:
            self.rank = None

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
            for i in range(len(cvs)):
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
        positions,
        velocities,
        forces,
        pe,
        ke,
        cell,
        cvs=None,
        rank=None,
    ):
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

    def close(self):
        """Close the trajectory file."""
        self.file.close()

    def early_close(self):
        """Close the trajectory file early, resizing datasets to the number of frames written."""
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
