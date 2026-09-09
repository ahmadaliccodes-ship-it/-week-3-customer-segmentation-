from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


def compare_k_values(X: pd.DataFrame | np.ndarray, k_values=range(2, 11), random_state=42, n_init=20) -> pd.DataFrame:
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km_labels = km.fit_predict(X)
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        agg_labels = agg.fit_predict(X)
        rows.append({
            "k": k,
            "kmeans_inertia": float(km.inertia_),
            "kmeans_silhouette": float(silhouette_score(X, km_labels)),
            "hier_silhouette": float(silhouette_score(X, agg_labels)),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X: pd.DataFrame | np.ndarray, k=2, random_state=42, n_init=20):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    return model, labels
