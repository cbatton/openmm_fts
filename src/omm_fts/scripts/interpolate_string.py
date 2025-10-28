"""Perform interpolation on string from simulation."""

import argparse
import glob
from typing import TYPE_CHECKING, Any

import h5py  # type: ignore[import-untyped]
import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from omm_fts.utils.natural_sort import natural_sort

if TYPE_CHECKING:
    from numpy.typing import NDArray


def main() -> None:
    """Perform interpolation on string from simulation."""
    parser = argparse.ArgumentParser(description="Run alanine dipeptide in vacuum")
    parser.add_argument(
        "--string_file",
        type=str,
        help="File to get string from",
        default="runs/0/ala2_0_string.h5",
    )
    parser.add_argument(
        "--string_file_new",
        type=str,
        help="File to get string from",
        default="ala2_interpolated_string.h5",
    )
    parser.add_argument(
        "--run_folder", type=str, help="Folder to get data from", default="runs"
    )
    parser.add_argument(
        "--num_points", type=int, help="Number of points to interpolate", default=64
    )

    args = parser.parse_args()
    string_file: str = args.string_file
    string_file_new: str = args.string_file_new
    run_folder: str = args.run_folder
    num_points: int = args.num_points

    with h5py.File(string_file, "r") as f:
        len_file = len(f.keys())
        string_cv: NDArray[Any] = f[f"config_{len_file - 1}/cvs"][:]

    num_points_original = string_cv.shape[0]
    t_spline = np.linspace(0, 1, num_points_original)
    string_spline = CubicSpline(t_spline, string_cv)

    t_new = np.linspace(0, 1, num_points)
    string_interpolated = string_spline(t_new)

    with h5py.File(string_file_new, "w") as f:
        f.create_dataset("cvs", data=string_interpolated)

    # now create initial positions for next simulation
    cv_data_list: list[NDArray[Any]] = []
    positions_list: list[NDArray[Any]] = []
    for i in range(num_points_original):
        files = natural_sort(glob.glob(f"{run_folder}/{i}/ala2_*_data.h5"))
        cv_data_: list[NDArray[Any]] = []
        positions_: list[NDArray[Any]] = []
        for file in files:
            with h5py.File(file, "r") as f_data:
                cv_0 = f_data["cv_0"][:]
                cv_1 = f_data["cv_1"][:]
                cvs = np.array([cv_0, cv_1]).T
                cvs = cvs.squeeze(0)
                cv_data_.append(cvs)
                positions_.append(f_data["positions"][:])
        cv_data_concat: NDArray[Any] = np.concatenate(cv_data_)
        positions_concat: NDArray[Any] = np.concatenate(positions_)
        cv_data_list.append(cv_data_concat)
        positions_list.append(positions_concat)
    cv_data: NDArray[Any] = np.array(cv_data_list)
    positions: NDArray[Any] = np.array(positions_list)
    cv_data = cv_data.reshape((-1, 2))
    positions = positions.reshape((-1, 22, 3))
    # now to find closest points to the interpolated string
    positions_initial: list[NDArray[Any]] = []
    for i in range(num_points):
        diff = np.linalg.norm(cv_data - string_interpolated[i], axis=1)
        index = np.argmin(diff)
        positions_initial.append(positions[index])

    positions_initial_array: NDArray[Any] = np.array(positions_initial)
    np.save("ala2_initial_positions.npy", positions_initial_array)


if __name__ == "__main__":
    main()
