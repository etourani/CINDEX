import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Load the csv data and calculate the average per shell
d = pd.read_csv('./dumpS_noavg_t2.csv')

# Select the relevant timestep and sort
d1 = d[d['step'] == 1].reset_index(drop=True)
d1 = d1.sort_values(by='atom_id').reset_index(drop=True)

# Shift positions so all coordinates are non-negative
positions = d1[['x', 'y', 'z']].copy()
x_shift = positions['x'].min()
y_shift = positions['y'].min()
z_shift = positions['z'].min()

positions['x'] -= x_shift
positions['y'] -= y_shift
positions['z'] -= z_shift

positions_array = positions[['x', 'y', 'z']].values

# Define periodic box size (a bit larger than max values)
dx = positions['x'].max() + 1e-3
dy = positions['y'].max() + 1e-3
dz = positions['z'].max() + 1e-3
box_size = (dx, dy, dz)

# Build the periodic KD-tree
tree = cKDTree(positions_array, boxsize=box_size)

# Define shells as (r_min, r_max) pairs
shell_ranges = [
    (0.3816, 0.7633), (0.7633, 1.145), (1.145, 1.5267),
    (1.5267, 1.908), (1.908, 2.29), (2.29, 2.672)
]

# Compute shell averages for each atom
for (r1, r2) in shell_ranges:
    shell_averages = []

    for pos in positions_array:
        outer_indices = tree.query_ball_point(pos, r2)
        inner_indices = tree.query_ball_point(pos, r1)
        shell_indices = list(set(outer_indices) - set(inner_indices))

        if shell_indices:
            shell_S_avg = d1.loc[shell_indices, 'S'].mean()
        else:
            shell_S_avg = np.nan

        shell_averages.append(shell_S_avg)
    
    col_name = f"S_ave_{r1:.1f}to{r2:.1f}"
    d1[col_name] = shell_averages


d1.to_csv('t2_averaged_S_shells_deltaR1point5.csv', index=False)

print(d1.head())
