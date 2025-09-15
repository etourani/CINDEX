from __future__ import annotations
import pandas as pd

DEFAULT_FEATURES = ["ent", "q6", "p2"]

def select_features(df: pd.DataFrame, features=None) -> pd.DataFrame:
    feats = features or DEFAULT_FEATURES
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[feats].copy()

def get_labels(df: pd.DataFrame, label_col: str = "Cluster_Label"):
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    y = df[label_col].values
    if set(pd.unique(y)) - {0, 1}:
        raise ValueError("Labels must be binary {0,1}.")
    return y
