# C-INDEX: TRAIN ON REPLICA-1, APPLY TO REPLICA-2 (argv-driven; no hard-coded paths)
import os, json, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mpl.rcParams['font.family'] = 'Nimbus Roman'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['grid.alpha'] = 0.10
mpl.rcParams['grid.linewidth'] = 0.8
mpl.rcParams['grid.linestyle'] = '--'

LABEL_COL      = "Cluster_Label"
RAW_FEATURES_3 = ["q6", "ent", "p2"]

S_THR      = -5.8
P2_THR     = 0.6
Q6_THR     = 0.18
C_PROB_THR = 0.5

COLS = dict(step="step", time="t", x="x", y="y", z="z", S="S", ent="ent", p2="p2", q6="q6")

TIME_SCALE = 4.7e-6
MARK_T_RAW = [1.05e7, 2.3e7]

def _sigmoid(z):
    import numpy as np
    z = np.clip(z, -700, 700)
    return 1.0 / (1.0 + np.exp(-z))

def _resolve_entropy_column(df):
    if COLS["ent"] in df.columns: return COLS["ent"]
    if COLS["S"] in df.columns:   return COLS["S"]
    raise ValueError("Could not find entropy column: expected 'ent' or 'S'.")

def _pick_x_axis(df):
    return COLS["time"] if COLS["time"] in df.columns else COLS["step"]

def train_and_export(train_csv: str, model_json: str):
    df = pd.read_csv(train_csv)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Training file must contain label column '{LABEL_COL}'")
    df[LABEL_COL] = df[LABEL_COL].replace(-1, 0).astype(int)
    for f in RAW_FEATURES_3:
        if f not in df.columns:
            raise ValueError(f"Training file missing feature '{f}'")
    X = df[RAW_FEATURES_3].copy()
    y = df[LABEL_COL].values
    pipe = Pipeline([('scale', StandardScaler()),
                     ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=0))])
    pipe.fit(X, y)
    y_prob = pipe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"[TRAIN] AUC3 on training snapshot: {auc:.6f}")
    scaler = pipe.named_steps['scale']; clf = pipe.named_steps['clf']
    model = {"feature_order": RAW_FEATURES_3,
             "scaler_mean": scaler.mean_.tolist(),
             "scaler_std": scaler.scale_.tolist(),
             "coef": clf.coef_[0].tolist(),
             "intercept": float(clf.intercept_[0]),
             "prob_threshold": C_PROB_THR,
             "positive_class": "crystal"}
    os.makedirs(os.path.dirname(model_json) or ".", exist_ok=True)
    with open(model_json, "w") as f: json.dump(model, f, indent=2)
    print(f"[TRAIN] Exported model to {model_json}")

def load_model(model_json: str):
    with open(model_json, "r") as f: m = json.load(f)
    assert len(m["feature_order"]) == 3 and len(m["scaler_mean"]) == 3 and len(m["scaler_std"]) == 3 and len(m["coef"]) == 3
    return m

def compute_cindex_on_replica2(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    import numpy as np
    ent_col = _resolve_entropy_column(df)
    Xcols_map = {"q6": COLS["q6"], "ent": ent_col, "p2": COLS["p2"]}
    X = np.vstack([df[Xcols_map[name]].to_numpy() for name in model["feature_order"]]).T
    means = np.array(model["scaler_mean"], dtype=float)
    stds  = np.array(model["scaler_std"], dtype=float)
    coefs = np.array(model["coef"], dtype=float)
    intercept = float(model["intercept"])
    Z = (X - means) / stds
    logit = intercept + Z.dot(coefs)
    prob  = _sigmoid(logit)
    ccls  = (prob >= model.get("prob_threshold", 0.5)).astype(np.uint8)
    out = df.copy()
    out["Cidx_logit"] = logit; out["Cidx_prob"]  = prob; out["Cidx_class"] = ccls
    return out

def cindex_diagnostics(df: pd.DataFrame, sample_csv: str, sample_n: int = 10):
    print("\\n[Diagnostics] Replica-2 (C-index)")
    if {"Cidx_logit","Cidx_prob"}.issubset(df.columns):
        print(f"  logit: [{df['Cidx_logit'].min():.3f}, {df['Cidx_logit'].max():.3f}]")
        print(f"  prob : [{df['Cidx_prob'].min():.3e}, {df['Cidx_prob'].max():.3e}]")
    if "Cidx_class" in df.columns:
        vc = df["Cidx_class"].value_counts(dropna=False).to_dict()
        print("  class counts:", vc)
    ent_col = COLS["ent"] if COLS["ent"] in df.columns else (COLS["S"] if COLS["S"] in df.columns else None)
    cols = [COLS.get("step"), COLS.get("time"), "atom_id", ent_col, COLS.get("p2"), COLS.get("q6"),
            "Cidx_logit", "Cidx_prob", "Cidx_class"]
    cols = [c for c in cols if c and c in df.columns]
    if cols:
        print("\\n[Head]\\n" + df[cols].head(sample_n).to_string(index=False))
        n = min(5000, len(df)); os.makedirs(os.path.dirname(sample_csv) or ".", exist_ok=True)
        df[cols].sample(n, random_state=1).to_csv(sample_csv, index=False)
        print(f"[Diagnostics] Wrote {n}-row sample to {sample_csv}")
    else:
        print("[Diagnostics] Skipped sample CSV (expected columns not found).")

def export_time_evolution_csv(df: pd.DataFrame, out_csv: str):
    if not (COLS["time"] in df.columns or COLS["step"] in df.columns):
        print("[Export] Skipped time evolution: no 't' or 'step'."); return pd.DataFrame()
    xcol = _pick_x_axis(df)
    ent_col = COLS["ent"] if COLS["ent"] in df.columns else (COLS["S"] if COLS["S"] in df.columns else None)
    if ent_col is None or not {"p2","q6","Cidx_class"}.issubset(set(df.columns)):
        print("[Export] Skipped time evolution: required columns missing."); return pd.DataFrame()
    S_thr    = (df[ent_col]       < S_THR).astype(np.uint8)
    p2_thr   = (df[COLS["p2"]]    > P2_THR).astype(np.uint8)
    q6_thr   = (df[COLS["q6"]]    > Q6_THR).astype(np.uint8)
    cidx_thr = (df["Cidx_class"]  == 1).astype(np.uint8)
    tmp = pd.DataFrame({xcol: df[xcol], "S_thr": S_thr, "p2_thr": p2_thr, "q6_thr": q6_thr, "Cidx_thr": cidx_thr})
    grp = tmp.groupby(xcol, sort=True).sum().reset_index()
    grp["t_scaled"] = grp[xcol] * TIME_SCALE
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    grp.to_csv(out_csv, index=False)
    print(f"[Export] Wrote time evolution (counts) -> {out_csv}")
    return grp

def plot_snapshot_compare(df: pd.DataFrame, fpath: str, snap_by_t=None, snap_by_step=None, max_points=500_000):
    if not all(c in df.columns for c in (COLS["x"], COLS["y"], COLS["z"])):
        print("[Figure A] Skipped: missing x,y,z."); return
    if (snap_by_t is None) and (snap_by_step is None):
        snap_by_t = df[COLS["time"]].iloc[0] if COLS["time"] in df.columns else None
        snap_by_step = df[COLS["step"]].iloc[0] if snap_by_t is None else None
    if snap_by_t is not None:
        if COLS["time"] not in df.columns: print("[Figure A] Skipped: no 't'."); return
        frame = df[df[COLS["time"]] == snap_by_t]; title_suffix = f"{COLS['time']}={snap_by_t}"
    else:
        if COLS["step"] not in df.columns: print("[Figure A] Skipped: no 'step'."); return
        frame = df[df[COLS["step"]] == snap_by_step]; title_suffix = f"{COLS['step']}={snap_by_step}"
    if frame.empty: print("[Figure A] Skipped: empty frame."); return
    if COLS["q6"] not in frame.columns or "Cidx_class" not in frame.columns:
        print("[Figure A] Skipped: missing q6 or Cidx_class."); return
    frame = frame.copy(); frame["q6_thr_class"] = (frame[COLS["q6"]] > Q6_THR).astype(np.uint8)
    if len(frame) > max_points: frame = frame.sample(max_points, random_state=42).copy()
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(frame[COLS["x"]], frame[COLS["y"]], frame[COLS["z"]], c=frame["Cidx_class"], s=1, alpha=0.8)
    ax1.set_title(f"C-index class — {title_suffix}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(frame[COLS["x"]], frame[COLS["y"]], frame[COLS["z"]], c=frame["q6_thr_class"], s=1, alpha=0.8)
    ax2.set_title(f"q6 > {Q6_THR} threshold — {title_suffix}")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    plt.tight_layout(); plt.savefig(fpath, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[Figure A] Saved -> {fpath}")

def plot_time_evolution_counts(df: pd.DataFrame, fpath: str):
    if not (COLS["time"] in df.columns or COLS["step"] in df.columns):
        print("[Figure B] Skipped: no 't' or 'step'."); return
    xcol = _pick_x_axis(df)
    ent_col = COLS["ent"] if COLS["ent"] in df.columns else (COLS["S"] if COLS["S"] in df.columns else None)
    if ent_col is None or not {"p2","q6","Cidx_class"}.issubset(set(df.columns)):
        print("[Figure B] Skipped: missing required columns."); return
    S_thr    = (df[ent_col]       < S_THR).astype(np.uint8)
    p2_thr   = (df[COLS["p2"]]    > P2_THR).astype(np.uint8)
    q6_thr   = (df[COLS["q6"]]    > Q6_THR).astype(np.uint8)
    cidx_thr = (df["Cidx_class"]  == 1).astype(np.uint8)
    tmp = pd.DataFrame({xcol: df[xcol], "S_thr": S_thr, "p2_thr": p2_thr, "q6_thr": q6_thr, "Cidx_thr": cidx_thr})
    grp = tmp.groupby(xcol, sort=True).sum().reset_index(); grp["t_scaled"] = grp[xcol] * TIME_SCALE
    plt.figure(figsize=(12, 8))
    plt.plot(grp["t_scaled"], grp["S_thr"],    label=f"{ent_col} < {S_THR}")
    plt.plot(grp["t_scaled"], grp["p2_thr"],   label=f"p2 > {P2_THR}")
    plt.plot(grp["t_scaled"], grp["q6_thr"],   label=f"q6 > {Q6_THR}")
    plt.plot(grp["t_scaled"], grp["Cidx_thr"], label="C-index = 1")
    for i, t0 in enumerate(MARK_T_RAW):
        ts = t0 * TIME_SCALE
        plt.axvline(ts, linestyle="--", linewidth=3.0, alpha=0.66)
        rng = (grp["t_scaled"].max() - grp["t_scaled"].min())
        plt.text(ts + 0.04*rng, plt.ylim()[0], f"$t_{{{i+1}}}$", ha="center", va="bottom", fontsize=16)
    plt.xlabel("Time (scaled)"); plt.ylabel("Number of atoms (thresholded)"); plt.title("Time evolution of thresholded counts")
    plt.legend(); plt.tight_layout(); plt.savefig(fpath, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[Figure B] Saved -> {fpath}")

def plot_time_evolution_fraction(df: pd.DataFrame, fpath: str, window=15):
    if not (COLS["time"] in df.columns or COLS["step"] in df.columns):
        print("[Figure B-fraction] Skipped: no 't' or 'step'."); return
    xcol = _pick_x_axis(df)
    ent_col = COLS["ent"] if COLS["ent"] in df.columns else (COLS["S"] if COLS["S"] in df.columns else None)
    if ent_col is None or not {"p2","q6","Cidx_class"}.issubset(set(df.columns)):
        print("[Figure B-fraction] Skipped: missing required columns."); return
    total = df.groupby(xcol)[xcol].count().rename("N").reset_index()
    S_thr    = (df[ent_col]       < S_THR).astype(np.uint8)
    p2_thr   = (df[COLS["p2"]]    > P2_THR).astype(np.uint8)
    q6_thr   = (df[COLS["q6"]]    > Q6_THR).astype(np.uint8)
    cidx_thr = (df["Cidx_class"]  == 1).astype(np.uint8)
    tmp = pd.DataFrame({xcol: df[xcol], "S_thr": S_thr, "p2_thr": p2_thr, "q6_thr": q6_thr, "Cidx_thr": cidx_thr})
    grp = tmp.groupby(xcol, sort=True).sum().reset_index(); grp = grp.merge(total, on=xcol, how="left")
    for col in ["S_thr","p2_thr","q6_thr","Cidx_thr"]:
        grp[col] = grp[col] / grp["N"]
    if window and window > 1:
        for col in ["S_thr","p2_thr","q6_thr","Cidx_thr"]:
            grp[col] = grp[col].rolling(window, min_periods=1, center=True).mean()
    grp["t_scaled"] = grp[xcol] * TIME_SCALE
    plt.figure(figsize=(12, 8))
    plt.plot(grp["t_scaled"], grp["S_thr"],    label=f"{ent_col} < {S_THR}")
    plt.plot(grp["t_scaled"], grp["p2_thr"],   label=f"p2 > {P2_THR}")
    plt.plot(grp["t_scaled"], grp["q6_thr"],   label=f"q6 > {Q6_THR}")
    plt.plot(grp["t_scaled"], grp["Cidx_thr"], label="C-index = 1")
    for i, t0 in enumerate(MARK_T_RAW):
        ts = t0 * TIME_SCALE
        plt.axvline(ts, linestyle="--", linewidth=3.0, alpha=0.66)
        rng = (grp["t_scaled"].max() - grp["t_scaled"].min())
        plt.text(ts + 0.04*rng, plt.ylim()[0], f"$t_{{{i+1}}}$", ha="center", va="bottom", fontsize=16)
    plt.xlabel("Time (scaled)"); plt.ylabel("Fraction of atoms"); plt.title("Time evolution of crystalline fraction")
    plt.legend(); plt.tight_layout(); plt.savefig(fpath, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[Figure B-fraction] Saved -> {fpath}")

def build_parser():
    p = argparse.ArgumentParser(description="Train on Replica-1 (q6, ent, p2) and apply to Replica-2 (q6, p2, S/ent).")
    p.add_argument("--data-dir", default="data", help="Base directory for CSVs if only names are given")
    p.add_argument("--train", help="Training CSV file NAME inside --data-dir (e.g. replica1_labeled.csv)")
    p.add_argument("--apply", help="Apply CSV file NAME inside --data-dir (e.g. replica2_all.csv)")
    p.add_argument("--train-csv", help="Full path to training CSV (overrides --train & --data-dir)")
    p.add_argument("--apply-csv", help="Full path to apply CSV (overrides --apply & --data-dir)")
    p.add_argument("--model-json", default="models/cindex_model_rep1.json")
    p.add_argument("--figA-out", default="figures/figA_snapshot_compare_cindex_vs_q6.png")
    p.add_argument("--figB-out", default="figures/figB_time_evolution_counts.png")
    p.add_argument("--figBfrac-out", default="figures/figB_time_evolution_fraction.png")
    p.add_argument("--sample-csv", default="docs/cindex_sample.csv")
    p.add_argument("--series-csv", default="docs/time_evolution_counts.csv")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--snap-by-t", type=float, default=None)
    p.add_argument("--snap-by-step", type=int, default=None)
    p.add_argument("--time-scale", type=float, default=TIME_SCALE)
    p.add_argument("--apply-label-col", default=None,
               help="If set and present in apply CSV, compute ROC/PR AUC (maps -1->0).")

    return p

def _resolve_inputs(args):
    train_csv = args.train_csv if args.train_csv else (os.path.join(args.data_dir, args.train) if args.train else None)
    apply_csv = args.apply_csv if args.apply_csv else (os.path.join(args.data_dir, args.apply) if args.apply else None)
    return train_csv, apply_csv

def main():
    args = build_parser().parse_args()
    global TIME_SCALE; TIME_SCALE = args.time_scale
    train_csv, apply_csv = _resolve_inputs(args)
    if not args.no_train:
        if not train_csv: raise SystemExit("Missing training CSV (use --train-csv OR --data-dir + --train)")
        print(f"[INFO] Training from: {train_csv}")
        train_and_export(train_csv, args.model_json)
    else:
        print("[INFO] Skipping training: using existing model.")
    if not apply_csv: raise SystemExit("Missing apply CSV (use --apply-csv OR --data-dir + --apply)")
    print(f"[INFO] Loading model: {args.model_json}")
    model = load_model(args.model_json)
    print(f"[INFO] Applying model to: {apply_csv}")
    df2 = pd.read_csv(apply_csv)
    df2 = compute_cindex_on_replica2(df2, model)
    from sklearn.metrics import roc_auc_score, average_precision_score
    if args.apply_label_col and args.apply_label_col in df2.columns:
        y = df2[args.apply_label_col].replace(-1, 0).astype(int).values
        s = df2["Cidx_prob"].values if "Cidx_prob" in df2.columns else df2["Cidx"].values
        roc = roc_auc_score(y, s)
        pr  = average_precision_score(y, s)
        print(f"[APPLY-AUC] ROC AUC={roc:.6f}  PR AUC={pr:.6f}  (n={len(df2)})")
    os.makedirs(os.path.dirname(args.sample_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.series_csv) or ".", exist_ok=True)
    cindex_diagnostics(df2, sample_csv=args.sample_csv)
    export_time_evolution_csv(df2, out_csv=args.series_csv)
    os.makedirs(os.path.dirname(args.figA_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.figB_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.figBfrac_out) or ".", exist_ok=True)
    #plot_snapshot_compare(df2, fpath=args.figA_out, snap_by_t=args.snap_by_t, snap_by_step=args.snap_by_step)
    plot_time_evolution_counts(df2, fpath=args.figB_out)
    plot_time_evolution_fraction(df2, fpath=args.figBfrac_out, window=15)

if __name__ == "__main__":
    main()
