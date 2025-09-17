from __future__ import annotations
import argparse
import sys
import pandas as pd

from .model import fit_logistic, save_model, load_model, predict_proba
from .features import select_features, get_labels
from .metrics import summarize_metrics, save_metrics, plot_curves
from .labelize_umap import umap_hdbscan_labelize, UMAPConfig, HDBSCANConfig
# NOTE: forward_select is imported lazily inside cmd_fs to avoid extra deps on non-FS commands


def cmd_labelize(a):
    df = pd.read_csv(a.input)

    if a.method == "umap-hdbscan":
        # REPLACE the old call with the next 17 lines:
        df_out, y, feats = umap_hdbscan_labelize(
            df,
            umap_cfg=UMAPConfig(
                n_neighbors=a.n_neighbors,
                min_dist=a.min_dist,
                n_components=2,
                metric=a.umap_metric,
                random_state=a.random_state,
            ),
            hdb_cfg=HDBSCANConfig(
                min_cluster_size=a.min_cluster_size,
                min_samples=a.min_samples,
                metric=a.hdb_metric,
            ),
            return_embedding=True,
            features=a.features,
        )
        out_col = a.out_col
        if out_col != "Cluster_Label":
            df_out.rename(columns={"Cluster_Label": out_col}, inplace=True)
        df_out.to_csv(a.output, index=False)
        print(f"[labelize] wrote: {a.output}  | features={feats} | method=umap-hdbscan")
    else:
        raise ValueError("Unknown method. Use --method umap-hdbscan")



def cmd_fs(a):
    # Lazy import so 'cindex labelize -h' doesn't require FS deps
    from .forward_select import forward_select_auc, FSConfig, dump_fs_results

    df = pd.read_csv(a.input)
    y = df[a.labels_col].astype(int)

    if a.features:
        feats = a.features
    else:
        # default: use all numeric predictors except label
        feats = [c for c in df.columns if c != a.labels_col and pd.api.types.is_numeric_dtype(df[c])]

    hist = forward_select_auc(
        X=df[feats],
        y=y,
        features=feats,
        cfg=FSConfig(
            estimator=a.estimator,
            n_splits=a.cv,
            random_state=a.random_state,
            max_features=a.max_features,
            early_delta=a.early_delta,
            early_patience=a.early_patience
        )
    )
    dump_fs_results(hist, out_json=a.report, out_csv=a.history_csv)
    print(f"[fs] best AUC={hist['auc'].max():.4f} | steps={len(hist)}")
    if a.history_csv:
        print(f"[fs] history CSV: {a.history_csv}")
    if a.report:
        print(f"[fs] JSON report: {a.report}")


def cmd_fit(a):
    df = pd.read_csv(a.train)
    pack = fit_logistic(
        df,
        label_col=a.class_col,
        features=a.features,
        penalty=a.penalty,
        C=a.C,
        solver=a.solver,
        max_iter=a.max_iter,
    )
    save_model(pack, a.model, notes=a.notes or None)
    print(f"Model saved to {a.model}")


def cmd_apply(a):
    df = pd.read_csv(a.input)
    pack = load_model(a.model)
    proba = predict_proba(df, pack)
    out = df.copy()
    out[a.proba_col] = proba
    out.to_csv(a.output, index=False)
    print(f"Scores written to {a.output}")


def cmd_validate(a):
    import numpy as np

    def _compute_and_report(y, yhat):
        metrics = summarize_metrics(y, yhat)
        if a.report:
            save_metrics(metrics, a.report)
        if a.plot:
            plot_curves(y, yhat, a.plot)
        print(metrics)

    # --- Mode A: labels.csv + scores.csv (no model needed) ---
    if a.labels_csv and a.scores_csv:
        labels_df = pd.read_csv(a.labels_csv)
        scores_df = pd.read_csv(a.scores_csv)

        if a.labels_col not in labels_df.columns:
            raise ValueError(f"Column '{a.labels_col}' not found in {a.labels_csv}")
        if a.score_col not in scores_df.columns:
            raise ValueError(f"Column '{a.score_col}' not found in {a.scores_csv}")

        y_series = labels_df[a.labels_col].replace(-1, 0).astype(int)
        s_series = pd.to_numeric(scores_df[a.score_col], errors="coerce")

        if a.key_cols:
            missing_l = [c for c in a.key_cols if c not in labels_df.columns]
            missing_s = [c for c in a.key_cols if c not in scores_df.columns]
            if missing_l or missing_s:
                raise ValueError(f"Missing merge keys; labels missing {missing_l}, scores missing {missing_s}")

            lsub = labels_df[a.key_cols + [a.labels_col]].copy()
            ssub = scores_df[a.key_cols + [a.score_col]].copy()
            merged = pd.merge(lsub, ssub, on=a.key_cols, how="inner")
            if merged.empty:
                raise ValueError("Merge on key columns produced no rows. Check keys/values.")
            y = merged[a.labels_col].replace(-1, 0).astype(int).values
            yhat = pd.to_numeric(merged[a.score_col], errors="coerce").fillna(0.0).values
        else:
            if len(y_series) != len(s_series):
                raise ValueError("labels and scores lengths differ; provide --key-cols to merge safely.")
            y = y_series.values
            yhat = s_series.fillna(0.0).values

        _compute_and_report(y, yhat)
        return

    # --- Mode B: model + test (legacy path) ---
    if a.model and a.test:
        df = pd.read_csv(a.test)
        if a.labels_col in df.columns:
            df[a.labels_col] = df[a.labels_col].replace(-1, 0).astype(int)
        pack = load_model(a.model)
        yhat = predict_proba(df, pack)
        y = get_labels(df, a.labels_col)
        _compute_and_report(y, yhat)
        return

    raise SystemExit(
        "validate: provide EITHER "
        "[--labels-csv PATH --scores-csv PATH (--key-cols ... optional)] "
        "OR [--model PATH --test PATH]."
    )

    # --- Mode B: model + test (existing behavior) ---
    if a.model and a.test:
        df = pd.read_csv(a.test)

        # Map -1 -> 0 for robustness (treat outlets/noise as melt)
        if a.labels_col in df.columns:
            df[a.labels_col] = df[a.labels_col].replace(-1, 0).astype(int)

        pack = load_model(a.model)
        yhat = predict_proba(df, pack)
        y = get_labels(df, a.labels_col)

        _compute_and_report(y, yhat)
        return

    # --- Neither mode satisfied ---
    raise SystemExit(
        "validate: provide EITHER "
        "[--labels-csv PATH --scores-csv PATH (--key-cols ... optional)] "
        "OR [--model PATH --test PATH]."
    )



def build_parser():
    parser = argparse.ArgumentParser(
        prog="cindex",
        description="C-index toolkit: labelize (UMAP+HDBSCAN), feature selection, fit/apply/validate."
    )
    sub = parser.add_subparsers(dest="cmd")
    try:
        sub.required = True
    except Exception:
        pass

    # --- labelize (umap-hdbscan) ---
    pl = sub.add_parser("labelize", help="Labelize via UMAP(2D) + HDBSCAN, then map to binary.")
    pl.add_argument("--input", required=True, help="Input CSV")
    pl.add_argument("--output", required=True, help="Output CSV (includes umap1,umap2, and label column)")
    pl.add_argument("--out-col", default="Cluster_Label", help="Name for the binary label column")
    pl.add_argument("--method", default="umap-hdbscan", choices=["umap-hdbscan"], help="Labeling method")
    pl.add_argument("--features", nargs="+", default=None,
                help="Explicit feature columns for UMAP; default: auto (ent/S, h, v, p2, all q*, all S_band*)")


    # UMAP params
    pl.add_argument("--n-neighbors", type=int, default=10)
    pl.add_argument("--min-dist", type=float, default=0.01)
    pl.add_argument("--umap-metric", default="manhattan")
    pl.add_argument("--random-state", type=int, default=11)

    # HDBSCAN params
    pl.add_argument("--min-cluster-size", type=int, default=180)
    pl.add_argument("--min-samples", type=int, default=80)
    pl.add_argument("--hdb-metric", default="euclidean")

    pl.set_defaults(func=cmd_labelize)

    # --- feature selection (forward) ---
    pfs = sub.add_parser("fs", help="Forward feature selection with GB or RF (ROC AUC).")
    pfs.add_argument("--input", required=True, help="Labeled CSV (must include labels column)")
    pfs.add_argument("--labels-col", default="Cluster_Label")
    pfs.add_argument("--features", nargs="+", default=None, help="Explicit feature list; default: all numeric except label")

    pfs.add_argument("--estimator", choices=["gb", "rf"], default="gb")
    pfs.add_argument("--cv", type=int, default=5)
    pfs.add_argument("--random-state", type=int, default=42)
    pfs.add_argument("--max-features", type=int, default=None, help="Stop after selecting this many features (optional)")
    pfs.add_argument("--early-delta", type=float, default=1e-4, help="Early stop if improvement < delta")
    pfs.add_argument("--early-patience", type=int, default=2)

    pfs.add_argument("--history-csv", default=None, help="Optional: write step-by-step history CSV")
    pfs.add_argument("--report", default=None, help="Optional: write JSON report (best AUC, features, history)")

    pfs.set_defaults(func=cmd_fs)

    # --- fit ---
    pf = sub.add_parser("fit", help="Train logistic C-index model")
    pf.add_argument("--train", required=True, help="CSV with features and labels")
    pf.add_argument("--model", required=True, help="Output YAML model path")
    pf.add_argument("--features", nargs="+", default=["ent", "q6", "p2"], help="Feature columns")
    pf.add_argument("--class-col", default="Cluster_Label", dest="class_col")
    pf.add_argument("--penalty", default="l2")
    pf.add_argument("--C", type=float, default=1.0)
    pf.add_argument("--solver", default="lbfgs")
    pf.add_argument("--max-iter", type=int, default=200, dest="max_iter")
    pf.add_argument("--notes", default=None)
    pf.set_defaults(func=cmd_fit)

    # --- apply ---
    pa = sub.add_parser("apply", help="Apply a trained model to feature CSV")
    pa.add_argument("--model", required=True, help="YAML model path")
    pa.add_argument("--input", required=True, help="Input CSV with features")
    pa.add_argument("--output", required=True, help="Output CSV with C-index scores")
    pa.add_argument("--proba-col", default="Cidx", dest="proba_col")
    pa.set_defaults(func=cmd_apply)

    # --- validate ---
    pv = sub.add_parser(
        "validate",
        help="Evaluate (A) external scores vs ground-truth labels OR (B) model on a test CSV."
    )

    # Mode A: labels + scores (no model needed)
    pv.add_argument("--labels-csv", default=None, help="CSV with ground-truth labels")
    pv.add_argument("--scores-csv", default=None, help="CSV with predicted scores")
    pv.add_argument("--labels-col", default="Cluster_Label", help="Column name for labels")
    pv.add_argument("--score-col", default="Cidx", help="Column name for predicted scores")
    pv.add_argument(
        "--key-cols", nargs="+", default=None,
        help="Optional merge keys present in both CSVs (e.g., atom_id t step). If omitted, align by row order."
    )

    # Mode B: model + test (legacy)
    pv.add_argument("--model", default=None, help="YAML model path")
    pv.add_argument("--test", default=None, help="CSV used for on-the-fly scoring")

    pv.add_argument("--report", default=None, help="Write JSON metrics (roc_auc, pr_auc)")
    pv.add_argument("--plot", default=None, help="Base name for ROC/PR PNGs (e.g., figures/curves.png)")
    pv.set_defaults(func=cmd_validate)



    return parser


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)
