"""Runs the Hamiltonian replica-exchange on alanine dipeptide in vacuum."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import openmm.unit as u

# Import MPI
from mpi4py import MPI
from openmm.openmm import CustomCVForce, CustomTorsionForce
from openmmtools import testsystems

from omm_fts.omm.omm_replica import OMMFFReplica


def main():
    """Run the Hamiltonian replica-exchange simulation."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Run MPI umbrella sampling simulation
    parser = argparse.ArgumentParser(description="Run alanine dipeptide in vacuum")
    parser.add_argument(
        "--integrator", type=str, help="Integrator to use", default="csvr_leapfrog"
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for the simulation", default=rank
    )
    parser.add_argument(
        "--string_file",
        type=str,
        help="String file to use",
        default="ala2_interpolated_string.h5",
    )
    parser.add_argument(
        "--num_data_points",
        type=int,
        help="Number of data points to collect",
        default=1000,
    )

    args = parser.parse_args()
    integrator = args.integrator
    seed = args.seed
    string_file = args.string_file
    num_data_points = args.num_data_points

    temperature = 300 * u.kelvin
    print(temperature)

    ala2 = testsystems.AlanineDipeptideImplicit()
    folder_name = f"runs_restart/{seed}/"
    file_name = f"{folder_name}ala2"
    if not Path(folder_name).exists():
        Path(folder_name).mkdir(parents=True)
    time_step = 0.002 * u.picoseconds
    if "csvr" in integrator:
        friction = 0.01 * u.picoseconds
    elif integrator == "langevin":
        friction = 0.5 / u.picoseconds
    elif integrator == "brownian":
        friction = 0.5 / u.picoseconds

    # Prepare custom force to give a backdoor to the CV, metric tensor
    cv0 = CustomTorsionForce("track*theta")
    cv0.addGlobalParameter("track", 0)
    cv0.addTorsion(4, 6, 8, 14)
    cv0.setForceGroup(16)

    cv1 = CustomTorsionForce("track*theta")
    cv1.addGlobalParameter("track", 0)
    cv1.addTorsion(6, 8, 14, 16)
    cv1.setForceGroup(17)

    # read values from restart
    with h5py.File(string_file, "r") as f:
        cvs = f["cvs"][:]
        phi0 = cvs[rank, 0] * u.radian
        psi0 = cvs[rank, 1] * u.radian

    # prepare more traditional biasing variables
    cv0_record = CustomTorsionForce("theta")
    cv0_record.addTorsion(4, 6, 8, 14)
    cv0_bias = CustomCVForce(
        "0.5 * kphi * delta^2; delta = min(min(abs(theta - phi0), abs(theta - phi0 + 2*pi)), abs(theta - phi0 - 2*pi)); pi=3.141592653589793"
    )
    kphi = 250 * u.kilojoules_per_mole / u.radian**2
    cv0_bias.addCollectiveVariable("theta", cv0_record)
    cv0_bias.addGlobalParameter("kphi", kphi)
    cv0_bias.addGlobalParameter("phi0", phi0)

    cv1_record = CustomTorsionForce("theta")
    cv1_record.addTorsion(6, 8, 14, 16)
    cv1_bias = CustomCVForce(
        "0.5 * kpsi * delta^2; delta = min(min(abs(theta - psi0), abs(theta - psi0 + 2*pi)), abs(theta - psi0 - 2*pi)); pi=3.141592653589793"
    )
    kpsi = 250 * u.kilojoules_per_mole / u.radian**2
    cv1_bias.addCollectiveVariable("theta", cv1_record)
    cv1_bias.addGlobalParameter("kpsi", kpsi)
    cv1_bias.addGlobalParameter("psi0", psi0)

    force_groups = [16, 17]
    parameter_name = ["phi0", "psi0"]
    force_parameter_name = ["kphi", "kpsi"]

    traj_init = np.load("ala2_initial_positions.npy")
    positions = traj_init[rank]
    ala2.positions = positions

    omm_ff = OMMFFReplica(
        ala2,
        platform="CPU",
        seed=seed + 1,
        folder_name=file_name,
        save_int=100,
        temperature=temperature,
        time_step=time_step,
        friction=friction,
        integrator_name=integrator,
        string_forces=[cv0, cv1],
        force_groups=force_groups,
        parameter_name=parameter_name,
        parameter_force_name=force_parameter_name,
        custom_forces=[cv0_bias, cv1_bias],
        comm=comm,
    )
    comm.Barrier()
    if rank == 0:
        print("Starting simulation")
    omm_ff.generate_long_trajectory(
        num_data_points=num_data_points,
        burn_in=0,
        save_freq=100,
        h5_freq=10,
        swap_freq=20,
    )


if __name__ == "__main__":
    main()
