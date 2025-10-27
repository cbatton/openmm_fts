"""Runs the finite-temperature string method on alanine dipeptide in vacuum."""

import argparse
from pathlib import Path

import openmm.unit as u

# Import MPI
from mpi4py import MPI
from openmm.openmm import CustomCVForce, CustomTorsionForce
from openmmtools import testsystems

from omm_fts.omm.omm_fts import OMMFF


def main():
    """Main function to run the finite-temperature string method on alanine dipeptide."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="Run alanine dipeptide in vacuum")
    parser.add_argument(
        "--integrator", type=str, help="Integrator to use", default="csvr_leapfrog"
    )
    parser.add_argument(
        "--seed", type=int, help="Seed for the simulation", default=rank
    )

    args = parser.parse_args()
    integrator = args.integrator
    seed = args.seed

    temperature = 300 * u.kelvin
    print(temperature)

    ala2 = testsystems.AlanineDipeptideImplicit()
    folder_name = f"runs/{seed}/"
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

    # prepare more traditional biasing variables
    cv0_record = CustomTorsionForce("theta")
    cv0_record.addTorsion(4, 6, 8, 14)
    cv0_bias = CustomCVForce(
        "0.5 * kphi * delta^2; delta = min(min(abs(theta - phi0), abs(theta - phi0 + 2*pi)), abs(theta - phi0 - 2*pi)); pi=3.141592653589793"
    )
    kphi = 250 * u.kilojoules_per_mole / u.radian**2
    phi0_start = -2.51 * u.radian
    phi0_end = 0.82 * u.radian
    phi0 = phi0_start + (phi0_end - phi0_start) * rank / (size - 1)
    cv0_bias.addCollectiveVariable("theta", cv0_record)
    cv0_bias.addGlobalParameter("kphi", kphi)
    cv0_bias.addGlobalParameter("phi0", phi0)

    cv1_record = CustomTorsionForce("theta")
    cv1_record.addTorsion(6, 8, 14, 16)
    cv1_bias = CustomCVForce(
        "0.5 * kpsi * delta^2; delta = min(min(abs(theta - psi0), abs(theta - psi0 + 2*pi)), abs(theta - psi0 - 2*pi)); pi=3.141592653589793"
    )
    kpsi = 250 * u.kilojoules_per_mole / u.radian**2
    psi0_start = 2.83 * u.radian
    psi0_end = -1.88 * u.radian
    psi0 = psi0_start + (psi0_end - psi0_start) * rank / (size - 1)
    cv1_bias.addCollectiveVariable("theta", cv1_record)
    cv1_bias.addGlobalParameter("kpsi", kpsi)
    cv1_bias.addGlobalParameter("psi0", psi0)

    force_groups = [16, 17]
    parameter_name = ["phi0", "psi0"]
    force_parameter_name = ["kphi", "kpsi"]

    omm_ff = OMMFF(
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
        string_freq=5,
        string_dt=0.1,
        string_kappa=0.05,
        cv_weights=[1.0, 1.0],
        update_ends=True,
        custom_forces=[cv0_bias, cv1_bias],
        comm=comm,
        minimize_init=True,
        minimize_intervals=40,
    )
    omm_ff.generate_long_trajectory(
        num_data_points=1000, burn_in=15, save_freq=100, h5_freq=10
    )


if __name__ == "__main__":
    main()
