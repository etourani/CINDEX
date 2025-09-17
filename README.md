# Supplementary Materials — C-index (Crystallinity Index) Toolkit

This archive accompanies the **C-index** paper "arXiv:2507.17980" and provides a reproducible toolkit to:
- **Train** a logistic-regression model (with `StandardScaler`) on a small, labeled snapshot
- **Apply** the learned parameters to larger systems (transfer under similar physics)
- **Validate** and reproduce paper figures/metrics

There is also additional tools in tools/ for calculations of OPs used in this paper. 

## Quick start
```bash
# 1) Install package locally
pip install -e ./
Optional, to check the install, you can try:
which cindex
cindex --help

# 2) Unsupervised labeling (UMAP→HDBSCAN)
cindex labelize --input data/examples/df_tmid1.csv --output data/examples/df_tmid1_labeled.csv
or if you need to change the hyperparamters through the command line:
cindex labelize \
  --input data/examples/df_tmid1.csv \
  --output data/examples/df_tmid1_labeled.csv \
  --features ent p2 q2 q4 q6 q8 q10 S_band1 S_band2 S_band3 S_band4 S_band5 S_band6 h v \
  --n-neighbors 10 \
  --min-dist 0.01 \
  --umap-metric manhattan \
  --random-state 11 \
  --min-cluster-size 180 \
  --min-samples 70 \
  --hdb-metric euclidean

Optionally, you can plot umap1 and umap2 with colors based on cluster labels (in the output file data/examples/df_tmid1_labeled.csv) to reproduce the same 2D UMAP space shown in the paper for the representative snapshot

# 3) Train on labeled data (set feature names if needed)
cindex fit --train data/examples/df_tmid1_labeled.csv --model models/cindex_model.yaml \
           --class-col Cluster_Label --features ent q6 p2

# 4) Apply to new data (a single snapshot or a trajectory file)(columns must match features used at training)
cindex apply --model models/cindex_model_tmid1.yaml --input data/examples/df_tss1.csv \
             --output data/examples/tss1_cindex_scores.csv

# Ground truth (for the higher dimension tss1 dataset) vs. produced scores 
cindex validate \
  --labels-csv data/examples/df_tss1_labeled.csv \
  --scores-csv data/examples/tss1_cindex_scores.csv \
  --labels-col Cluster_Label \
  --score-col Cidx \
  --key-cols atom_id \
  --report results/metrics_tss1.json \
  --plot results/tss1_curves.png


# 6) Optional: Feature selection (forward; GB or RF)
cindex fs --input data/examples/train_features.csv --labels-col Cluster_Label \
          --estimator gb --history-csv docs/fs_history.csv \
          --report docs/fs_report.json
```

> **Feature note:** Package defaults use columns **`S, q6, p2`**.  
> The **example script** (below) demonstrates training on **`q6, ent, p2`** (where `ent` is an entropy column) and
> applying to data with either `S` *or* `ent`. Use whichever maps to the workflow.

---

## Example script (argv only, no hard-coded paths)
Train on Replica‑1 (using `q6, ent, p2` and label), apply to Replica‑2 (with `q6, p2, `ent`), and export Cluster_Labels for the unseen dataset and related figures.

```bash
python code/scripts/replica_train_apply.py   --data-dir data/examples   --train replica1_labeled.csv   --apply replica2_all.csv   --model-json models/cindex_model_rep1.json   --figA-out figures/figA_snapshot_compare_cindex_vs_q6.png   --figB-out figures/figB_time_evolution_counts.png   --figBfrac-out figures/figB_time_evolution_fraction.png   --sample-csv docs/cindex_sample.csv   --series-csv docs/time_evolution_counts.csv
```

### Alternative: pass full paths
```bash
python code/scripts/replica_train_apply.py   --train-csv /abs/path/to/replica1_labeled.csv   --apply-csv /abs/path/to/replica2_all.csv   --model-json models/cindex_model_rep1.json
```

### Output overview
- JSON model with scaler stats + logistic coefficients
- Snapshot comparison plot (C-index vs. `q6` threshold)
- Time-evolution plots (counts and fractions)
- Sample rows and per-frame series CSVs
