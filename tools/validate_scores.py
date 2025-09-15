import argparse, json
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def main():
    ap = argparse.ArgumentParser(description="Compute AUC from an apply() scores CSV + labeled CSV.")
    ap.add_argument("--scores", required=True, help="CSV from `cindex apply` (must contain Cidx column)")
    ap.add_argument("--labels-csv", required=True, help="CSV with ground-truth labels")
    ap.add_argument("--labels-col", default="Cluster_Label", help="Ground-truth column name")
    ap.add_argument("--score-col", default="Cidx", help="Predicted score column name in --scores")
    ap.add_argument("--key-cols", nargs="*", default=None, help="Join keys (e.g. atom_id step). If omitted, align by row index.")
    ap.add_argument("--report", default=None, help="Optional JSON metrics output path")
    args = ap.parse_args()

    df_s = pd.read_csv(args.scores)
    df_y = pd.read_csv(args.labels_csv)

    if args.key_cols:
        for k in args.key_cols:
            if k not in df_s.columns or k not in df_y.columns:
                raise SystemExit(f"Key '{k}' not found in both files.")
        merged = pd.merge(df_s, df_y[[*args.key_cols, args.labels_col]], on=args.key_cols, how="inner", validate="one_to_one")
    else:
        if len(df_s) != len(df_y):
            raise SystemExit("Row counts differ and no --key-cols provided; cannot align.")
        merged = df_s.copy()
        merged[args.labels_col] = df_y[args.labels_col].values

    if args.score_col not in merged.columns:
        raise SystemExit(f"Score column '{args.score_col}' not in scores CSV.")
    if args.labels_col not in merged.columns:
        raise SystemExit(f"Labels column '{args.labels_col}' not in labels CSV.")

    y = merged[args.labels_col].replace(-1, 0).astype(int).values
    s = merged[args.score_col].astype(float).values

    roc = float(roc_auc_score(y, s))
    pr  = float(average_precision_score(y, s))
    out = {"roc_auc": roc, "pr_auc": pr, "n": int(len(merged))}
    print(out)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
