from __future__ import annotations

import pandas as pd
from scipy.stats import f_oneway

SPEND_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def cluster_profile(df: pd.DataFrame, cluster_col="cluster_k2") -> dict:
    sizes = df[cluster_col].value_counts().sort_index()
    pct = sizes / len(df) * 100
    means = df.groupby(cluster_col)[SPEND_COLS].mean()
    medians = df.groupby(cluster_col)[SPEND_COLS].median()
    separation = []
    for col in SPEND_COLS:
        groups = [g[col].values for _, g in df.groupby(cluster_col)]
        f_stat, p_val = f_oneway(*groups)
        separation.append({"Feature": col, "F-statistic": float(f_stat), "P-value": float(p_val)})
    channel_pct = pd.crosstab(df[cluster_col], df["Channel"].map({1: "Horeca", 2: "Retail"}), normalize="index") * 100
    region_pct = pd.crosstab(df[cluster_col], df["Region"], normalize="index") * 100
    return {"sizes": sizes, "pct": pct, "means": means, "medians": medians, "separation": pd.DataFrame(separation), "channel_pct": channel_pct, "region_pct": region_pct}
