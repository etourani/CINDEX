from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan


@dataclass
class UMAPConfig:
    n_neighbors: int = 10
    min_dist: float = 0.01
    n_components: int = 2
    metric: str = "manhattan"
    random_state: int = 50


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 180
    min_samples: Optional[int] = 70
    metric: str = "euclidean"
    #leaf_size: int = 15



def _detect_bands(df: pd.DataFrame) -> List[str]:
    # Any column named S_band* (order-preserving)
    bands = sorted([c for c in df.columns if c.startswith("S_band")],
                   key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or "0"))
    return bands


# Impute any missing (NaN) values in the entropy bands
def _impute_bands_mean_neighbor(df: pd.DataFrame, band_cols: List[str]) -> None:
    """Impute NaNs for S_band columns using neighbor means (first/last: single neighbor)."""
    if not band_cols:
        return
    for i, col in enumerate(band_cols):
        if i == 0 and len(band_cols) > 1:
            nxt = band_cols[i+1]
            df.loc[df[col].isna(), col] = df[nxt]
        elif i == len(band_cols) - 1 and len(band_cols) > 1:
            prv = band_cols[i-1]
            df.loc[df[col].isna(), col] = df[prv]
        elif 0 < i < len(band_cols) - 1:
            prv, nxt = band_cols[i-1], band_cols[i+1]
            df.loc[df[col].isna(), col] = df[[prv, nxt]].mean(axis=1)
    # If any remain NaN (e.g., whole row missing), final fallback to column mean
    for col in band_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].mean(), inplace=True)



# --- add near the top ---
import re

def _normalize_columns(df: pd.DataFrame) -> None:
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
    )

def _gather_features(df: pd.DataFrame, force_features: Optional[List[str]] = None
                     ) -> Tuple[pd.DataFrame, List[str]]:
    """If force_features is provided, use them (after normalization & numeric coercion).
       Else, auto: ent/S, h, v, p2, all q*, all S_band* (with S_band imputation).
    """
    _normalize_columns(df)

    # If user provided explicit list, use it verbatim (present & normalized)
    if force_features:
        ordered = [c for c in force_features if c in df.columns]
        if not ordered:
            raise ValueError(f"No requested features found in CSV: {force_features}")
        X = df[ordered].copy()
    else:
        cols: List[str] = []
        ent_like = "ent" if "ent" in df.columns else ("S" if "S" in df.columns else None)
        if ent_like:
            cols.append(ent_like)
        for opt in ("h", "v", "p2", "q6"):
            if opt in df.columns:
                cols.append(opt)

        # any q* (q2,q4,q6,q8,q10,...)
        qcols = [c for c in df.columns if c.startswith("q")]
        # keep order but avoid dup q6
        for qc in qcols:
            if qc not in cols:
                cols.append(qc)

        # S_band*
        band_cols = _detect_bands(df)
        _impute_bands_mean_neighbor(df, band_cols)
        cols += band_cols

        # dedupe, preserve order
        seen, ordered = set(), []
        for c in cols:
            if c in df.columns and c not in seen:
                seen.add(c); ordered.append(c)

        if not ordered:
            raise ValueError("No usable features found (expected ent/S, p2, q*, S_band*).")
        X = df[ordered].copy()

    # Force numeric; median-impute any NaNs introduced by coercion
    X = X.apply(pd.to_numeric, errors="coerce")
    for c in X.columns:
        if X[c].isna().any():
            X[c].fillna(X[c].median(), inplace=True)

    return X, list(X.columns)



def _map_clusters_to_binary(df: pd.DataFrame,
                            labels: np.ndarray,
                            feats_used: List[str]) -> np.ndarray:
    """Map HDBSCAN labels -> binary {0,1}.
       Heuristic: cluster with larger mean q6 (or lower mean ent/S) → crystal (1).
       Noise (-1) assigned to nearest non-noise cluster centroid in UMAP space if possible,
       otherwise to majority class.
    """
    y = np.zeros_like(labels, dtype=int)
    unique = sorted(set(labels) - {-1})
    if not unique:
        warnings.warn("HDBSCAN found only noise. Assigning all zeros.")
        return y

    # Decide crystal cluster via q6 or ent/S
    ref_col = "q6" if "q6" in df.columns else None
    ent_col = "ent" if "ent" in df.columns else ("S" if "S" in df.columns else None)

    def score_cluster(k: int) -> float:
        idx = (labels == k)
        if ref_col is not None:
            return df.loc[idx, ref_col].mean()  # higher → more crystalline
        elif ent_col is not None:
            return -df.loc[idx, ent_col].mean()  # lower ent/S → more crystalline
        else:
            return df.loc[idx, feats_used].mean().mean()  # fallback

    scores = {k: score_cluster(k) for k in unique}
    crystal_id = max(scores, key=scores.get)
    amorph_id_candidates = [k for k in unique if k != crystal_id]
    amorph_id = amorph_id_candidates[0] if amorph_id_candidates else None

    # Assign 1 to crystal cluster, 0 to others
    y[(labels == crystal_id)] = 1
    if amorph_id is not None:
        y[(labels == amorph_id)] = 0

    # Noise handling: attach to nearest centroid in UMAP (if we have umap1/umap2)
    if {"umap1", "umap2"}.issubset(df.columns) and len(unique) > 0:
        centers = {k: df.loc[labels == k, ["umap1", "umap2"]].mean().values for k in unique}
        noise_idx = np.where(labels == -1)[0]
        if len(noise_idx) > 0:
            pts = df.loc[noise_idx, ["umap1", "umap2"]].values
            for i, pt in zip(noise_idx, pts):
                # nearest center
                k_star = min(centers, key=lambda k: np.linalg.norm(pt - centers[k]))
                y[i] = 1 if k_star == crystal_id else 0
    else:
        # Fallback: majority class
        if (labels != -1).any():
            maj_is_crystal = (y[labels != -1].mean() >= 0.5)
            y[labels == -1] = 1 if maj_is_crystal else 0

    return y


def umap_hdbscan_labelize(
    df: pd.DataFrame,
    umap_cfg: UMAPConfig = UMAPConfig(),
    hdb_cfg: HDBSCANConfig = HDBSCANConfig(),
    return_embedding: bool = True,
    features: Optional[List[str]] = None,   # <-- NEW
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    X, feats = _gather_features(df, force_features=features)
    """Main routine: scale --> UMAP(2d) --> HDBSCAN --> binary labels.

    Returns: (df_out, y, feats_used)
      - df_out includes 'umap1','umap2' and 'Cluster_Label' (binary)
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    reducer = umap.UMAP(
        n_neighbors=umap_cfg.n_neighbors,
        min_dist=umap_cfg.min_dist,
        n_components=umap_cfg.n_components,
        metric=umap_cfg.metric,
        random_state=umap_cfg.random_state,
    )
    emb = reducer.fit_transform(Xs)  # (n,2)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdb_cfg.min_cluster_size,
        min_samples=hdb_cfg.min_samples,
        metric=hdb_cfg.metric
    )
    raw_labels = clusterer.fit_predict(emb)

    out = df.copy()
    out["umap1"], out["umap2"] = emb[:, 0], emb[:, 1]
    y = _map_clusters_to_binary(out, raw_labels, feats)
    out["Cluster_Label"] = y
    return out, y, feats
