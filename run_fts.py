import argparse
import glob
import re
from pathlib import Path

import mdtraj as md
import numpy as np
import openmm.unit as u

# Import MPI
from mpi4py import MPI
from openmm.openmm import CustomCVForce, CustomTorsionForce
from openmmtools import testsystems

from omm import OMMFF


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Run MPI umbrella sampling simulation
parser = argparse.ArgumentParser(description="Run alanine dipeptide in vacuum")
parser.add_argument(
    "--integrator", type=str, help="Integrator to use", default="csvr_leapfrog"
)
parser.add_argument("--opes_type", type=str, help="Type of OPES", default="OPES_METAD")
parser.add_argument("--seed", type=int, help="Seed for the simulation", default=rank)

args = parser.parse_args()
integrator = args.integrator
opes_type = args.opes_type
seed = args.seed

if opes_type not in ["OPES_METAD", "OPES_METAD_EXPLORE"]:
    raise ValueError("OPES type must be OPES_METAD or OPES_METAD_EXPLORE")

# Have Ar units for WCA
temperature = 300 * u.kelvin
print(temperature)

ala2 = testsystems.AlanineDipeptideImplicit(constraints=None)
folder_name = f"runs/{seed}/"
file_name = f"{folder_name}ala2"
folder_name_walker = f"{folder_name}walkers/"
if not Path(folder_name).exists():
    Path(folder_name).mkdir(parents=True)
if not Path(folder_name_walker).exists():
    Path(folder_name_walker).mkdir(parents=True)
time_step = 0.0005 * u.picoseconds
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
# phi0_start = -3 * u.radian
phi0_start = -2.51 * u.radian
phi0_end = 0.82 * u.radian
phi0 = phi0_start + (phi0_end - phi0_start) * rank / (size - 1)
# if rank == 0:
# phi0 = -3.00 * u.radian
cv0_bias.addCollectiveVariable("theta", cv0_record)
cv0_bias.addGlobalParameter("kphi", kphi)
cv0_bias.addGlobalParameter("phi0", phi0)

cv1_record = CustomTorsionForce("theta")
cv1_record.addTorsion(6, 8, 14, 16)
cv1_bias = CustomCVForce(
    "0.5 * kpsi * delta^2; delta = min(min(abs(theta - psi0), abs(theta - psi0 + 2*pi)), abs(theta - psi0 - 2*pi)); pi=3.141592653589793"
)
kpsi = 250 * u.kilojoules_per_mole / u.radian**2
# psi0_start = -3 * u.radian
psi0_start = 2.83 * u.radian
psi0_end = -1.88 * u.radian
psi0 = psi0_start + (psi0_end - psi0_start) * rank / (size - 1)
# if rank == 0:
# psi0 = -3.00 * u.radian
cv1_bias.addCollectiveVariable("theta", cv1_record)
cv1_bias.addGlobalParameter("kpsi", kpsi)
cv1_bias.addGlobalParameter("psi0", psi0)

force_groups = [16, 17]
parameter_name = ["phi0", "psi0"]

# find initial positions with closest phi and psi
traj_files = natural_sort(glob.glob("runs_ref_metad/0/ala2_*.h5"))
traj_files = [x for x in traj_files if "data" not in x][:-1]
psi_angles = []
phi_angles = []
traj_file_identity = []
for i, traj_file in enumerate(traj_files):
    traj = md.load(traj_file)
    phi = md.compute_phi(traj)
    psi = md.compute_psi(traj)
    phi_angles.append(phi[1])
    psi_angles.append(psi[1])
    traj_file_identity.append(i * np.ones(len(phi[1])))
psi_angles = np.concatenate(psi_angles)
phi_angles = np.concatenate(phi_angles)
traj_file_identity = np.concatenate(traj_file_identity)

# get the file with the closest phi and psi
min_dist = 1000
for i in range(len(phi_angles)):
    dist = np.sqrt(
        (phi_angles[i] - phi0.value_in_unit(u.radian)) ** 2
        + (psi_angles[i] - psi0.value_in_unit(u.radian)) ** 2
    )
    if dist < min_dist:
        min_dist = dist
        traj_file = traj_file_identity[i]

# get the initial positions
traj = md.load(traj_files[int(traj_file)])
phi = md.compute_phi(traj)[1]
psi = md.compute_psi(traj)[1]
# find min index in this traj
min_dist = 1000
min_index = 0
min_phi = 0
min_psi = 0
for i in range(len(phi)):
    dist = np.sqrt(
        (phi[i] - phi0.value_in_unit(u.radian)) ** 2
        + (psi[i] - psi0.value_in_unit(u.radian)) ** 2
    )
    if dist < min_dist:
        min_dist = dist
        min_index = i
        min_phi = phi[i]
        min_psi = psi[i]
if rank == 0:
    print(min_dist, min_phi, min_psi, phi0, psi0)
positions = traj.xyz[min_index]
ala2.positions = positions
print(ala2.positions)

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
    string_freq=20,
    string_dt=0.1,
    string_kappa=0.1,
    cv_weights=[1.0, 1.0],
    update_ends=False,
    custom_forces=[cv0_bias, cv1_bias],
    comm=comm,
)
omm_ff.generate_long_trajectory(
    num_data_points=1000, burn_in=15, save_freq=100, h5_freq=10
)
