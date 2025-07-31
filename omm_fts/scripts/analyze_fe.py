"""Perform free energy analysis on OpenMM simulations."""

import argparse
import glob

import h5py
import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline

from omm_fts.utils.natural_sort import natural_sort


def main():
    """Perform free energy analysis on OpenMM simulations."""
    parser = argparse.ArgumentParser(description="Run alanine dipeptide in vacuum")
    parser.add_argument(
        "--string_file",
        type=str,
        help="String file to use",
        default="ala2_interpolated_string.h5",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        help="Burn-in period for the analysis",
        default=10,
    )
    parser.add_argument(
        "--num_bootstrap",
        type=int,
        help="Number of bootstrap samples to use",
        default=1000,
    )

    args = parser.parse_args()
    string_file = args.string_file
    burn_in = args.burn_in
    num_bootstrap = args.num_bootstrap
    num_bootstrap += 1

    # now to load in files
    with h5py.File(string_file, "r") as f:
        cvs_string = f["cvs"][:]

    num_folders = cvs_string.shape[0]
    cv_data = []
    rank = []
    for i in range(num_folders):
        files = natural_sort(glob.glob(f"runs_restart/{i}/ala2_*_data.h5"))
        cv_data_ = []
        rank_ = []
        for file in files[burn_in:]:
            with h5py.File(file, "r") as f_data:
                cv_0 = f_data["cv_0"][:]
                cv_1 = f_data["cv_1"][:]
                cvs = np.array([cv_0, cv_1]).T
                cvs = cvs.squeeze(0)
                cv_data_.append(cvs)
                rank_.append(f_data["rank"][:])
        cv_data_ = np.concatenate(cv_data_)
        rank_ = np.concatenate(rank_)
        cv_data.append(cv_data_)
        rank.append(rank_)

    cv_data = np.array(cv_data)
    rank = np.array(rank)
    cv_data = cv_data.reshape(-1, 2)
    rank = rank.reshape(-1)

    cvs_average = np.zeros((num_folders, 2))
    for i in range(num_folders):
        cvs_average[i, :] = np.mean(cv_data[rank == i, :], axis=0)

    # free energy along collective variables
    k_values = np.array([250, 250])
    df_dz = k_values * (cvs_string - cvs_average)

    # now to obtain the free energy along the string
    t_spline = np.linspace(0, 1, cvs_string.shape[0])
    cvs_spline = CubicSpline(t_spline, cvs_string)
    cvs_spline_prime = cvs_spline.derivative()
    cvs_prime = cvs_spline_prime(t_spline)
    df_ds = df_dz * cvs_prime
    df_ds = np.sum(df_ds, axis=1)
    integral = cumulative_simpson(y=df_ds, x=t_spline)
    integral = np.append(0, integral)

    # save to h5 file
    with h5py.File("ala2_free_energy.h5", "w") as f:
        f.create_dataset("cvs", data=cvs_string)
        f.create_dataset("cvs_average", data=cvs_average)
        f.create_dataset("df_dz", data=df_dz)
        f.create_dataset("df_ds", data=df_ds)
        f.create_dataset("fe", data=integral)
        f.create_dataset("t_spline", data=t_spline)


if __name__ == "__main__":
    main()
