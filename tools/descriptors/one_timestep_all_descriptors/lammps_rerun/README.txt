README.txt
===========

This directory contains scripts and data for computing individual per-particle descriptors 
using LAMMPS `rerun` mode in a minimal format. Each descriptor is calculated 
independently in its corresponding subdirectory.

Purpose and Scope:
------------------

- To ensure clarity and reusability for any reader, we provide a clean and minimal demonstration 
  for each descriptor that depends on LAMMPS post-processing.
- These examples are meant to be educational and replicable, even for users unfamiliar with the full workflow.

Demonstration Example:
----------------------

- This folder specifically includes a demonstration of descriptor computation for snapshot **t2 of Replica 2**, 
  as referenced in **Section S5.2** of the Supplementary Material of the paper.

- The following descriptors are included:
  - **entropy/** – scalar averaged entropy per particle (`\bar S_i`)
  - **enthalpy/** – local averaged enthalpy values
  - **voro/** – Voronoi volume (or related geometric descriptors)
  - **ent_bands/** – entropy band averages

- Associated input files:
  - `c1s1run1_300K.data`: data file used in LAMMPS rerun
  - `dump_t2`: LAMMPS dump file corresponding to the t2 snapshot

Broader Context:
----------------

- While this folder only shows the descriptor computation for one representative snapshot (t2),
  the **same scripts and methods** were applied to compute descriptors across **10 representative snapshots** 
  from **Replica 1**, used in dimensionality reduction, C-index definition, and machine learning tasks.

- The computed results for those snapshots are included elsewhere in the dataset (e.g., under `dim_red/`, 
  `cindex/`, or other relevant analysis folders), where they appear as `.csv` files for further processing.

Notes:
------

- Each descriptor-specific folder contains its own README and Python scripts where applicable.
- The workflow is designed to run independently from the main simulation, relying on `dump` and `data` files only.
