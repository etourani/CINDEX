from __future__ import annotations
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .features import select_features

@dataclass
class LRPack:
    type: str
    features: list
    coef: list
    intercept: float
    scaler_mean: list
    scaler_scale: list

    def to_dict(self):
        return {
            "model": {
                "type": self.type,
                "features": self.features,
                "coef": self.coef,
                "intercept": self.intercept,
                "scaler": {"mean": self.scaler_mean, "scale": self.scaler_scale},
            }
        }

    @staticmethod
    def from_dict(d):
        m = d["model"]
        s = m.get("scaler", {})
        return LRPack(
            type=m["type"],
            features=m["features"],
            coef=m["coef"],
            intercept=m["intercept"],
            scaler_mean=s.get("mean", [0.0]*len(m["features"])),
            scaler_scale=s.get("scale", [1.0]*len(m["features"])),
        )

def fit_logistic(df: pd.DataFrame, label_col="y", features=None,
                 penalty="l2", C=1.0, solver="lbfgs", max_iter=200):
    X = select_features(df, features).values
    y = df[label_col].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
    clf.fit(Xs, y)

    pack = LRPack(
        type="logistic_regression",
        features=features or ["ent", "q6", "p2"],
        coef=clf.coef_.ravel().tolist(),
        intercept=float(clf.intercept_.ravel()[0]),
        scaler_mean=scaler.mean_.ravel().tolist(),
        scaler_scale=scaler.scale_.ravel().tolist(),
    )
    return pack

def predict_proba(df: pd.DataFrame, pack: LRPack):
    X = select_features(df, pack.features).values
    Xs = (X - np.array(pack.scaler_mean)) / np.array(pack.scaler_scale)
    z = Xs.dot(np.array(pack.coef)) + pack.intercept
    proba = 1.0 / (1.0 + np.exp(-z))
    return proba

def save_model(pack: LRPack, path: str, notes: str = None):
    d = pack.to_dict()
    if notes:
        d["notes"] = notes
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)

def load_model(path: str) -> LRPack:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return LRPack.from_dict(d)
