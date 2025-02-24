import argparse
from pathlib import Path

import numpy as np
import openmm.unit as u

# Import MPI
from mpi4py import MPI
from openmm.openmm import CustomBondForce, CustomCVForce
from openmm.unit import nanometers
from openmmtools import testsystems
from openmmplumed import PlumedForce

from omm import OMMFF

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Run MPI umbrella sampling simulation
parser = argparse.ArgumentParser(description="Run alanine dipeptide in vacuum")
parser.add_argument(
    "--integrator", type=str, help="Integrator to use", default="csvr_leapfrog"
)
parser.add_argument(
    "--opes_type", type=str, help="Type of OPES", default="OPES_METAD"
)
parser.add_argument(
    "--seed", type=int, help="Seed for the simulation", default=rank
)

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
time_step = 0.001 * u.picoseconds
if "csvr" in integrator:
    friction = 0.01 * u.picoseconds
elif integrator == "langevin":
    friction = 0.5 / u.picoseconds
elif integrator == "brownian":
    friction = 0.5 / u.picoseconds

# Prepare custom force to give a backdoor to the CV
# index0_1 = 0
# index0_2 = 1
# cv0 = CustomBondForce("r")
# cv0.addBond(index0_1, index0_2)
# r0 = 0.0 * nanometers
# k0 = 0.0 * epsilon / sigma**2
# bondForce0 = CustomCVForce("0.5 * k0 * (cv0-r0)^2")
# bondForce0.addGlobalParameter("k0", k0)
# bondForce0.addGlobalParameter("r0", r0)
# bondForce0.addCollectiveVariable("cv0", cv0)

# setup OPES
# cv_line = "cv: DISTANCE ATOMS=1,2\n"
# opes_line = f"opes: {opes_type} ...\n"
# opes_line += "ARG=cv\n"
# barrier = 12.0*0.824*120*u.kelvin*kB
# convert barrier to kJ/mol
# barrier = 1.25*barrier.value_in_unit(u.kilojoules_per_mole)
# print(barrier)
# opes_line += f"BARRIER={barrier}\n"
# opes_line += f"PACE=1000 TEMP={temperature.value_in_unit(u.kelvin)}\n"
# opes_line += f"FILE={folder_name_walker}/kernels.data\n"
# opes_line += f"STATE_WFILE={folder_name_walker}/state.data\n"
# opes_line += "STATE_WSTRIDE=10000\n"
# opes_line += "NLIST\n"
# opes_line += "...\n"
# opes_line += f"PRINT STRIDE=1000 ARG=cv,opes.* FILE={folder_name_walker}/COLVAR\n"

# plumed_file = open(f"{folder_name_walker}plumed.dat", "w")
# plumed_file.write("FLUSH STRIDE=10\n")
# plumed_file.write(cv_line)
# plumed_file.write(opes_line)
# plumed_file.close()
# Read in plumed file as a string
# plumed_script = open(f"{folder_name_walker}plumed.dat", "r").read()
# print(plumed_script)
# plumed_force = PlumedForce(plumed_script)

omm_ff = OMMFF(
    ala2,
    platform="CUDA",
    seed=seed+1,
    folder_name=file_name,
    save_int=1000,
    temperature=temperature,
    time_step=time_step,
    friction=friction,
    integrator_name=integrator,
    # custom_forces=[bondForce0],
    # plumed_force=plumed_force,
)
omm_ff.generate_long_trajectory(num_data_points=200000)
