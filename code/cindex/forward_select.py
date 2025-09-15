from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class FSConfig:
    estimator: str = "gb"   # "gb" or "rf"
    n_splits: int = 5
    random_state: int = 11
    max_features: Optional[int] = None
    early_delta: float = 1e-4  # stop if improvement < delta for 2 steps
    early_patience: int = 2


def _make_estimator(kind: str, random_state: int):
    if kind == "gb":
        est = GradientBoostingClassifier(random_state=random_state)
    elif kind == "rf":
        est = RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state
        )
    else:
        raise ValueError("Unknown estimator (use 'gb' or 'rf').")
    # pipeline with scaler to be safe for GB (RF ignores)
    pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("clf", est)])
    return pipe


def forward_select_auc(
    X: pd.DataFrame,
    y: pd.Series,
    features: Optional[List[str]] = None,
    cfg: FSConfig = FSConfig()
) -> pd.DataFrame:
    feats = features or list(X.columns)
    remaining = feats.copy()
    selected: List[str] = []

    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    scorer = make_scorer(roc_auc_score, needs_threshold=True)

    history = []
    best_auc_so_far = 0.0
    patience = cfg.early_patience

    max_steps = cfg.max_features or len(remaining)
    for step in range(max_steps):
        # Evaluate each remaining feature when appended
        scores: List[Tuple[str, float]] = []
        for f in remaining:
            cand = selected + [f]
            est = _make_estimator(cfg.estimator, cfg.random_state)
            auc = cross_val_score(est, X[cand], y, scoring=scorer, cv=cv, n_jobs=None).mean()
            scores.append((f, float(auc)))

        f_star, auc_star = max(scores, key=lambda t: t[1])
        selected.append(f_star)
        remaining.remove(f_star)
        history.append({"step": step + 1, "added": f_star, "auc": auc_star, "selected": selected.copy()})

        # early stopping (plateau)
        if auc_star - best_auc_so_far < cfg.early_delta:
            patience -= 1
            if patience <= 0:
                break
        else:
            best_auc_so_far = auc_star
            patience = cfg.early_patience

    return pd.DataFrame(history)


def dump_fs_results(df_hist: pd.DataFrame, out_json: Optional[str], out_csv: Optional[str]) -> None:
    if out_csv:
        df_hist.to_csv(out_csv, index=False)
    if out_json:
        payload: Dict[str, object] = {
            "best_auc": float(df_hist["auc"].max()),
            "best_step": int(df_hist["auc"].idxmax()) + 1,
            "best_features": df_hist.loc[df_hist["auc"].idxmax(), "selected"],
            "history": df_hist.to_dict(orient="records"),
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

