import argparse
import glob
import re

import gsd
import gsd.hoomd
import mdtraj as md
import numpy as np


def boolean_string(s):
    if s not in {"False", "True", "false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "True" or s == "true"


# Load in h5 files, convert to gsd
parser = argparse.ArgumentParser(description="Convert .h5 files to .gsd")
parser.add_argument("--files_to_convert", type=str, help="Files to convert")
parser.add_argument(
    "--include_last", type=boolean_string, help="Include last file", default=False
)

args = parser.parse_args()
files_to_convert = args.files_to_convert
include_last = args.include_last

# gather all files that end with h5 and don't have "data" in them
files = glob.glob(f"{files_to_convert}*.h5")
files = [file for file in files if "data" not in file]


# perform a natural sort
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


files = natural_sort(files)
print(files)
if not include_last:
    files = files[:-1]
print(files)

# get the number of atoms and unit cell
traj = md.load(files[0])

n_atoms = traj.n_atoms
unit_cell = traj.unitcell_lengths[0]

# get positions
positions = []

for file in files:
    traj = md.load(file)
    positions.append(traj.xyz)

positions = np.concatenate(positions, axis=0)

# get elements
elements = []

traj = md.load(files[0])
elements_2 = []
for atom in traj.topology.atoms:
    elements_2.append(str(atom.element.symbol))

# get unique elements
elements = list(set(elements_2))

# get bonds
bonds = [[bond[0].index, bond[1].index] for bond in traj.topology.bonds]
bonds = np.array(bonds)

positions *= 10  # convert to angstroms
unit_cell *= 10

# apply PBC to positions
# get how much shifted
positions -= np.rint(positions / unit_cell) * unit_cell

# save to gsd
gsd_file = gsd.hoomd.open("trajectory.gsd", "w")

def create_frame(i, n_atoms, positions, elements, elements_2, unit_cell, bonds):
    frame = gsd.hoomd.Frame()
    frame.configuration.step = i
    frame.configuration.dimensions = 3
    frame.particles.N = n_atoms
    frame.particles.position = positions

    frame.particles.types = elements
    frame.particles.typeid = [elements.index(element) for element in elements_2]

    # create bonds
    frame.bonds.N = len(bonds)
    frame.bonds.group = bonds
    frame.bonds.types = ["bond"]
    frame.bonds.typeid = [0 for i in range(len(bonds))]

    frame.configuration.box = [unit_cell[0], unit_cell[1], unit_cell[2], 0, 0, 0]

    return frame


for i in range(positions.shape[0]):
    frame = create_frame(i, n_atoms, positions[i], elements, elements_2, unit_cell, bonds)
    gsd_file.append(frame)

gsd_file.close()
