import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


d = pd.read_csv('./dumpH_t2.csv')

d1 = d[d['step'] == 1].reset_index(drop=True)
print('d1 columns: ', d1.columns)
print('d1: ', d1)

positions = d1[['x', 'y', 'z']].values


dx = d1['x'].max() + 1e-3
dy = d1['y'].max() + 1e-3
dz = d1['z'].max() + 1e-3
box_size = (dx, dy, dz)
print(dx, dy, dz)
print('xmax: ', d1['x'].max())
print('ymax: ', d1['y'].max())
print('zmax: ', d1['z'].max())

tree = cKDTree(positions, boxsize=box_size)

cutoff = 2.

h_ave = []

for i, pos in enumerate(positions):
    # Find indices of points within the cutoff_radius
    indices = tree.query_ball_point(pos, cutoff)

    # Calculate the average h including the atom itself
    h_avg = d1.loc[indices, 'h'].mean()
    h_ave.append(h_avg)

# Add the average h values as a new column to the dataframe
d1['h_ave'] = h_ave

print(d1[['atom_id', 'h', 'h_ave']])

d1.to_csv('averaged_H_t2.csv', index=False)
