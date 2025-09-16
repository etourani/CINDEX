# p2_all_steps.py
import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------- helpers (same math/logic as yours) -----------------
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

def process_one_step(step_value, df_step, k=60):
    """
    EXACT SAME LOGIC as the single-step script, just wrapped for one step.
    Uses the column names: atom_id, mol_id, S.
    """
    ds = df_step.copy()
    ds["_orig_idx"] = np.arange(len(ds))

    # bounds from the first row of this step (same logic)
    xlo = ds['xlo'].iloc[0]; xhi = ds['xhi'].iloc[0]
    ylo = ds['ylo'].iloc[0]; yhi = ds['yhi'].iloc[0]
    zlo = ds['zlo'].iloc[0]; zhi = ds['zhi'].iloc[0]

    # sort by atom_id as you did (aid previously)
    ds_sorted = ds.sort_values(by=['atom_id'])

    # neighbor atoms in chain (prev/next) per mol (mol_id previously mol)
    ds_sorted['x_next'] = ds_sorted.groupby('mol_id')['x'].shift(-1)
    ds_sorted['y_next'] = ds_sorted.groupby('mol_id')['y'].shift(-1)
    ds_sorted['z_next'] = ds_sorted.groupby('mol_id')['z'].shift(-1)

    ds_sorted['x_prev'] = ds_sorted.groupby('mol_id')['x'].shift(1)
    ds_sorted['y_prev'] = ds_sorted.groupby('mol_id')['y'].shift(1)
    ds_sorted['z_prev'] = ds_sorted.groupby('mol_id')['z'].shift(1)

    ds = ds_sorted.copy()

    # same "mid_next"/"mid_prev" via mol_id column
    ds['mid_next'] = ds.groupby('mol_id')['mol_id'].shift(-1)
    ds['mid_prev'] = ds.groupby('mol_id')['mol_id'].shift(1)

    # chain tangent vector only if both neighbors exist AND same mol
    same_both = (ds['mol_id'] == ds['mid_next']) & (ds['mol_id'] == ds['mid_prev'])
    ds['vector_x'] = np.where(same_both, ds['x_next'] - ds['x_prev'], np.nan)
    ds['vector_y'] = np.where(same_both, ds['y_next'] - ds['y_prev'], np.nan)
    ds['vector_z'] = np.where(same_both, ds['z_next'] - ds['z_prev'], np.nan)

    # coordinate shift by per-step minima (same logic)
    ds['x'] -= ds['x'].min()
    ds['y'] -= ds['y'].min()
    ds['z'] -= ds['z'].min()

    # fill vectors NaN with 0 before neighbor search (same behavior; no FutureWarning)
    ds[['vector_x','vector_y','vector_z']] = (
        ds[['vector_x','vector_y','vector_z']].fillna(0).infer_objects(copy=False)
    )

    # boxsize + tiny padding (same)
    box_size = (xhi - xlo + 0.02, yhi - ylo + 0.02, zhi - zlo + 0.02)

    coords = ds[['x', 'y', 'z']].values
    tree = cKDTree(coords, boxsize=box_size)
    k_eff = min(k, len(coords))
    _, indices = tree.query(coords, k=k_eff)

    if indices.ndim == 1:
        indices = indices[:, None]

    vectors = ds[['vector_x', 'vector_y', 'vector_z']].values
    aids = ds['atom_id'].values          # same logic as aids
    mids = ds['mol_id'].values           # same logic as mids

    p2_values = []
    for i in range(len(coords)):
        vec_i = vectors[i]
        aids_i = aids[i]
        mid_i = mids[i]
        p2_list = []
        for j in indices[i]:
            if i != j:
                aids_j = aids[j]
                mid_j = mids[j]
                # same exclusion rule: different chain OR |delta atom_id| > 4
                if (mid_i != mid_j) or (abs(aids_i - aids_j) > 4):
                    vec_j = vectors[j]
                    _, cos_theta = calculate_angle_vectors(vec_i, vec_j)
                    p2 = compute_p2(cos_theta)
                    p2_list.append(p2)
        mean_p2 = np.mean(p2_list) if p2_list else np.nan
        p2_values.append(round_to_precision(mean_p2))

    ds['p2'] = pd.Series(p2_values, index=ds.index).fillna(0)

    # restore original order; drop temps created inside this function
    ds = ds.sort_values(by=['_orig_idx']).drop(
        columns=['_orig_idx','x_prev','y_prev','z_prev','x_next','y_next','z_next','mid_prev','mid_next'],
        errors='ignore'
    )

    print(f"Step {step_value} done! (N={len(ds)})")
    return ds

# ----------------- main (streaming writes + fsync) -----------------
def main():
    ap = argparse.ArgumentParser(description="Compute p2 for ALL timesteps (streaming writes).")
    ap.add_argument("input_file", help="Input CSV (has header row)")
    ap.add_argument("output_file", help="Output CSV path")
    ap.add_argument("--workers", type=int, default=1, help="Parallel processes over steps")
    ap.add_argument("--k", type=int, default=60, help="k neighbors (same as original)")
    args = ap.parse_args()

    # Read WITH header row; specify dtypes to avoid DtypeWarning
    df = pd.read_csv(
        args.input_file,
        sep=',',
        header=0,  # your file has a header
        low_memory=False,
        dtype={
            'step':'int64','t':'int64','atom_id':'int64',
            'mol_id':'float64','x':'float64','y':'float64','z':'float64',
            'S':'float64','xlo':'float64','xhi':'float64','ylo':'float64',
            'yhi':'float64','zlo':'float64','zhi':'float64'
        }
    )

    # Split by step
    step_groups = list(df.groupby('step', sort=True))

    # Prepare output (streamed appends)
    out_path = args.output_file
    if os.path.exists(out_path):
        os.remove(out_path)
    header_written = False

    def _finalize_and_write(df_step):
        nonlocal header_written  # declare before using
        # Drop helper columns you don't want in the final CSV
        drop_cols = [
            'xlo','xhi','ylo','yhi','zlo','zhi',
            'vector_x','vector_y','vector_z',
            'x_prev','y_prev','z_prev','x_next','y_next','z_next',
            'mid_prev','mid_next','_orig_idx'
        ]
        df_step = df_step.drop(columns=[c for c in drop_cols if c in df_step.columns])

        # Append and force to disk
        mode = 'a'
        write_header = not header_written
        with open(out_path, mode, buffering=1) as f:
            df_step.to_csv(f, index=False, header=write_header)
            f.flush()
            os.fsync(f.fileno())
        header_written = True

    # Process and write step-by-step
    if args.workers > 1 and len(step_groups) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one_step, int(step), g.copy(), args.k)
                    for step, g in step_groups]
            for fut in as_completed(futs):
                df_step = fut.result()
                _finalize_and_write(df_step)
    else:
        for step, g in step_groups:
            df_step = process_one_step(int(step), g.copy(), args.k)
            _finalize_and_write(df_step)

    print(f"Writing to: {out_path}")
    print("All steps completed.")

if __name__ == "__main__":
    main()
