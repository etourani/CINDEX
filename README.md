# C-index (Crystallinity Index) Toolkit — Supplementary Materials

**C-index** is a lightweight, reproducible workflow for polymer crystallization analysis from MD data.  
It trains a **logistic model** on a single labeled snapshot using routinely available descriptors (typically **S/ent**, **p2**, **q6**) and then **applies the frozen scaler + coefficients** to any timestep or larger system with similar physics—avoiding repeated high-dimensional DR/clustering.

This repository accompanies the paper **arXiv:2507.17980** and provides:
- **Labeling** (UMAP --> HDBSCAN) for one snapshot  
- **Training** a logistic C-index 
- **Applying** the frozen model to new snapshots/trajectories  
- **Validation** ROC/PR, reports, etc  
- Helper scripts for computing/handling descriptors (see `scripts/`, `tools/`)

---

## Quick start

```bash
# 1) Install
pip install -e .
# sanity check
which cindex
python -c "import cindex, importlib; importlib.import_module('cindex.cli')"
cindex --help
```

### 2) Unsupervised labeling (UMAP --> HDBSCAN)

```bash
# Minimal
cindex labelize   --input data/examples/df_tmid1.csv   --output data/examples/df_tmid1_labeled.csv

# With explicit hyperparameters and feature list
cindex labelize   --input data/examples/df_tmid1.csv   --output data/examples/df_tmid1_labeled.csv   --features ent p2 q2 q4 q6 q8 q10 S_band1 S_band2 S_band3 S_band4 S_band5 S_band6 h v   --n-neighbors 10   --min-dist 0.01   --umap-metric manhattan   --random-state 11   --min-cluster-size 180   --min-samples 70   --hdb-metric euclidean
```

> Tip: after labeling, you can plot `umap1` vs `umap2` colored by the output cluster labels to reproduce the 2D UMAP view from the paper.

### 3) Train on labeled data

```bash
cindex fit   --train data/examples/df_tmid1_labeled.csv   --model models/cindex_model.yaml   --class-col Cluster_Label   --features ent q6 p2     # use 'S q6 p2' if your column is named S instead of ent
```

### 4) Apply to new data (single snapshot or trajectory)

```bash
cindex apply   --model models/cindex_model.yaml   --input data/examples/df_tss1.csv   --output data/examples/tss1_cindex_scores.csv
```

### 5) Validate (ground truth vs scores)

```bash
cindex validate   --labels-csv data/examples/df_tss1_labeled.csv   --scores-csv data/examples/tss1_cindex_scores.csv   --labels-col Cluster_Label   --score-col Cidx   --key-cols atom_id   --report results/metrics_tss1.json   --plot results/tss1_curves.png
```

### 6) Optional: Forward feature selection (GB or RF)

```bash
cindex fs   --input data/examples/df_tmid1_labeled.csv   --labels-col Cluster_Label   --estimator gb   --history-csv docs/fs_history.csv   --report docs/fs_report.json
```

---


**Expected artifacts:**
- `models/cindex_model.yaml` — frozen scaler stats + logistic regression coefficients  
- `data/examples/tss1_cindex_scores.csv` — per-atom C-index scores for the evaluation data  
- `results/metrics_tss1.json`, `results/tss1_curves.png` — ROC/PR metrics and curves

---


## Another example: Reproduce **SM §S5.2** — Transferability demo (train once, apply everywhere)

This reproduces the “train-on-one-snapshot, apply across time/system size” experiment.

**Inputs (examples included):**
- `data/examples/df_tmid1_labeled.csv` — a representative snapshot for labeled from Replica 1  
- `data/examples/replica2_stride20.csv` — an independent snapshot/trajectory to evaluate from Replica 2. Stride20: one snapshot was selected for every 20 to reduce the file size for upload.


python scripts/replica_train_apply.py   --data-dir data/examples   --train df_tmid1_labeled.csv   --apply replica2_stride20.csv   --model-json models/cindex_model_rep1.json   --figA-out figures/figA_snapshot_compare_cindex_vs_q6.png   --figB-out figures/figB_time_evolution_counts.png   --figBfrac-out figures/figB_time_evolution_fraction.png   --sample-csv docs/cindex_sample.csv   --series-csv docs/time_evolution_counts.csv


**Outputs**
- Frozen model (scaler stats + logistic coefficients) in JSON/YAML
- Time-evolution plots (counts and fractions)
- Sample rows and per-frame series CSVs

---


**How to cite**

If you use this software, the C-index order parameter, please cite:

C-index paper
Machine Learning Workflow for Analysis of High-Dimensional Order Parameter Space: A Case Study of Polymer Crystallization from Molecular Dynamics Simulations.
Tourani, E.; Edwards, B. J.; Khomami, B.
arXiv:2507.17980 (2025). https://doi.org/10.48550/arXiv.2507.17980

**Related work & companion repositories**
DEB (Directional Entropy Bands) — descriptors for interfacial structure/crystallization:
https://github.com/etourani/DEB
If you use the entropy bands or DEB features from this toolkit, please also cite the DEB paper.

---

**License**

See LICENSE. For academic and non-commercial use; please contact the authors for other licensing options.

---

**Changelog / contributing / conduct**

See CHANGELOG.md for version history.

Contributions are welcome—please open an issue or PR (see CONTRIBUTING.md).

We follow the project’s CODE_OF_CONDUCT.md.

