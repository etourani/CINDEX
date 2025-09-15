# Example usage:
#   python p2.py './traj_files/dumpS_t2.csv' './p2_results/ent_p2_t2.csv'
#
# Here, `dumpS_t2.csv` is the input trajectory (already converted to CSV).
# The script calculates the p2 values and appends them to the same CSV,
# producing an output file containing both entropy (`S`) and p2 values.
#
# Note:
## - The example uses `dumpS_t2.csv` for convenience, but any dump CSV can be used.
#   To generate such a CSV from a LAMMPS trajectory dump file, use `to_csv.py`
#   from the `lammps_rerun` folder.
# - p2 values are not computed for head and tail particles in each chain.
#   Later, when combining descriptors into a unified dataset, these missing values
#   will be imputed using the average of the lower quartile, as discussed in the main paper.
#


import sys
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
import os

# Check for the required arguments
if len(sys.argv) < 3:
    print("Usage: python script.py <input_file> <output_directory>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

def calculate_angle_vectors(vec_i, vec_j):
    dot_product = np.dot(vec_i, vec_j)
    norm_i = np.linalg.norm(vec_i)
    norm_j = np.linalg.norm(vec_j)
    cos_theta = dot_product / (norm_i * norm_j) if norm_i != 0 and norm_j != 0 else 0
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle), cos_theta

def round_to_precision(value, precision=4):
    return round(value, precision) if not np.isnan(value) else np.nan

def compute_p2(cos_theta):
    return round_to_precision((3 * cos_theta**2 - 1) / 2)



columns = ['step', 't', 'aid', 'mol', 'x', 'y', 'z', 'ent', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi']
df = pd.read_csv(input_file, sep=',', names=columns, header=None)#.drop(columns=['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'])


print(df)
#print(df[df['step'] == 5])
#ds = df.copy() # for input files with one dt
ds = df[df['step'] == 1].copy() # for input files with multiple dts

print(ds)

xlo = ds['xlo'].iloc[0]
xhi = ds['xhi'].iloc[0]
ylo = ds['ylo'].iloc[0]
yhi = ds['yhi'].iloc[0]
zlo = ds['zlo'].iloc[0]
zhi = ds['zhi'].iloc[0]

ds_sorted = ds.sort_values(by=['aid'])

ds_sorted['x_next'] = ds_sorted.groupby('mol')['x'].shift(-1)
ds_sorted['y_next'] = ds_sorted.groupby('mol')['y'].shift(-1)
ds_sorted['z_next'] = ds_sorted.groupby('mol')['z'].shift(-1)

ds_sorted['x_prev'] = ds_sorted.groupby('mol')['x'].shift(1)
ds_sorted['y_prev'] = ds_sorted.groupby('mol')['y'].shift(1)
ds_sorted['z_prev'] = ds_sorted.groupby('mol')['z'].shift(1)

ds = ds_sorted.copy()

ds['mid_next'] = ds.groupby('mol')['mol'].shift(-1)
ds['mid_prev'] = ds.groupby('mol')['mol'].shift(1)


ds['vector_x'] = np.where((ds['mol'] == ds['mid_next']) & (ds['mol'] == ds['mid_prev']), ds['x_next'] - ds['x_prev'], np.nan)
ds['vector_y'] = np.where((ds['mol'] == ds['mid_next']) & (ds['mol'] == ds['mid_prev']), ds['y_next'] - ds['y_prev'], np.nan)
ds['vector_z'] = np.where((ds['mol'] == ds['mid_next']) & (ds['mol'] == ds['mid_prev']), ds['z_next'] - ds['z_prev'], np.nan)

# scipy.cKDTree uses [0, L) as the range to apply periodicity, so I do ds['x'] -= ds['x'].min() etc:
ds['x'] -= ds['x'].min()
ds['y'] -= ds['y'].min()
ds['z'] -= ds['z'].min()


ds[['vector_x', 'vector_y', 'vector_z']] = ds[['vector_x', 'vector_y', 'vector_z']].fillna(0) # because nan may effect k = 50


box_size = (xhi-xlo + 2e-2, yhi-ylo + 2e-2, zhi-zlo + 2e-2)
print(f"Box size: {box_size}")
coords = ds[['x', 'y', 'z']].values
tree = cKDTree(coords, boxsize=box_size)
_, indices = tree.query(coords, k=60)  # For the neighbor list size, please see Fig. S3(c)

vectors = ds[['vector_x', 'vector_y', 'vector_z']].values
p2_values = []

for i in range(len(coords)):
    vec_i = vectors[i]
    aids_i = ds.iloc[i]['aid']
    mid_i = ds.iloc[i]['mol']
    p2_list = []

    for j in indices[i]:
        if i != j:
            aids_j = ds.iloc[j]['aid']
            mid_j = ds.iloc[j]['mol']
            # same exclusion rule: different chain OR |delta atom_id| > 4
            if mid_i != mid_j or abs(aids_i - aids_j) > 4: 
                vec_j = vectors[j]
                angle, cos_theta = calculate_angle_vectors(vec_i, vec_j)
                p2 = compute_p2(cos_theta)
                p2_list.append(p2)

    mean_p2 = np.mean(p2_list) if p2_list else np.nan
    rounded_mean_p2 = round_to_precision(mean_p2)
    p2_values.append(rounded_mean_p2)

ds['p2'] = p2_values
print(f'Step {ds["step"].iloc[0]} done!')
print(ds)

ds['p2'] = ds['p2'].fillna(0)
print(ds)
ds.to_csv(output_file, index=False)



