import argparse
from pathlib import Path

import openmm.unit as u

# Import MPI
from mpi4py import MPI
from openmm.openmm import CustomTorsionForce
from openmmplumed import PlumedForce
from openmmtools import testsystems

from omm import OMMFF

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
folder_name = f"runs_opes/{seed}/"
file_name = f"{folder_name}ala2"
folder_name_walker = f"{folder_name}walkers/"
if not Path(folder_name).exists():
    Path(folder_name).mkdir(parents=True)
if not Path(folder_name_walker).exists():
    Path(folder_name_walker).mkdir(parents=True)
time_step = 0.001 * u.picoseconds
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

cv2 = CustomTorsionForce("track*theta")
cv2.addGlobalParameter("track", 0)
cv2.addTorsion(5, 4, 6, 8)
cv2.setForceGroup(18)

cv3 = CustomTorsionForce("track*theta")
cv3.addGlobalParameter("track", 0)
cv3.addTorsion(8, 14, 16, 17)
cv3.setForceGroup(19)

force_groups = [16, 17, 18, 19]

# setup OPES
cv_line = "phi: TORSION ATOMS=5,7,9,15\n"
cv_line += "psi: TORSION ATOMS=7,9,15,17\n"
cv_line += "theta: TORSION ATOMS=6,5,7,9\n"
cv_line += "zeta: TORSION ATOMS=9,15,17,18\n"
opes_line = f"opes: {opes_type} ...\n"
opes_line += "ARG=phi,psi,theta,zeta\n"
opes_line += "BARRIER=30\n"
opes_line += f"PACE=500 TEMP={temperature.value_in_unit(u.kelvin)}\n"
opes_line += f"FILE={folder_name_walker}/kernels.data\n"
opes_line += f"STATE_WFILE={folder_name_walker}/state.data\n"
opes_line += "STATE_WSTRIDE=10000\n"
opes_line += "NLIST\n"
opes_line += "...\n"
opes_line += f"PRINT STRIDE=1000 ARG=phi,psi,theta,zeta,opes.* FILE={folder_name_walker}/COLVAR\n"

plumed_file = open(f"{folder_name_walker}plumed.dat", "w")
plumed_file.write("FLUSH STRIDE=10\n")
plumed_file.write(cv_line)
plumed_file.write(opes_line)
plumed_file.close()
# Read in plumed file as a string
plumed_script = open(f"{folder_name_walker}plumed.dat", "r").read()
# print(plumed_script)
plumed_force = PlumedForce(plumed_script)

omm_ff = OMMFF(
    ala2,
    platform="CUDA",
    seed=seed + 1,
    folder_name=file_name,
    save_int=1000,
    temperature=temperature,
    time_step=time_step,
    friction=friction,
    integrator_name=integrator,
    custom_forces=[cv0, cv1],
    force_groups=force_groups,
    plumed_force=plumed_force,
)
omm_ff.generate_long_trajectory(num_data_points=200000, time_max=8 * 60 * 60)
