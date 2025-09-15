# boo stands for bond orientational order parameter (BOO parameter) 
# example usage woiuld be "python boo.py './traj_files/dump_t2' 2300" in which ./traj_files/dump_t2 is the dump file from lammps and 2300 is the desired timestep in that traj file. 


import sys
from pyscal3 import System, Trajectory
import matplotlib.pyplot as plt
import numpy as np
import csv

# Retrieve the argument from the command line
if len(sys.argv) < 3:
    print("Usage: python boo.py <trajectory_path> <index_or_timestep>")
    sys.exit(1)

trajectory_path = sys.argv[1]
index = int(sys.argv[2])

traj = Trajectory(trajectory_path)
slides = traj[:]    # slice the entire trajectory
syss = slides.to_system()  # list-like of pyscal3 Systems
sys_at_index = syss[2]



sys_at_index.find.neighbors(method='cutoff', cutoff=1.45)

q2 = sys_at_index.calculate.steinhardt_parameter(2, averaged=True)[0]
q4 = sys_at_index.calculate.steinhardt_parameter(4, averaged=True)[0]
q6 = sys_at_index.calculate.steinhardt_parameter(6, averaged=True)[0]
q8 = sys_at_index.calculate.steinhardt_parameter(8, averaged=True)[0]
q10 = sys_at_index.calculate.steinhardt_parameter(10, averaged=True)[0]

filename = f'./boo_results/boo_{index}.csv'
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['atom_id', 'q2', 'q4', 'q6', 'q8', 'q10'])

    for i, atom_id in enumerate(sys_at_index.atoms.ids):
        writer.writerow([atom_id, q2[i], q4[i], q6[i], q8[i], q10[i]])

print(f'Qi data written to {filename}')


