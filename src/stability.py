from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def seed_stability(X, k, seeds=(42, 7, 123, 2024, 99), n_init=20) -> float:
    labels = [KMeans(n_clusters=k, random_state=s, n_init=n_init).fit_predict(X) for s in seeds]
    aris = [adjusted_rand_score(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    return float(np.mean(aris))


def bootstrap_stability(X, k, n_bootstrap=50, random_state=12345, model_random_state=42, n_init=20) -> list[float]:
    X = np.asarray(X)
    baseline = KMeans(n_clusters=k, random_state=model_random_state, n_init=n_init).fit(X)
    baseline_labels = baseline.labels_
    rng = np.random.RandomState(random_state)
    values = []
    n = len(X)
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(np.arange(n), size=n, replace=True)
        oob = np.setdiff1d(np.arange(n), np.unique(boot_idx))
        if len(oob) < max(10, 2 * k):
            continue
        model = KMeans(n_clusters=k, random_state=model_random_state, n_init=n_init).fit(X[boot_idx])
        values.append(float(adjusted_rand_score(baseline_labels[oob], model.predict(X[oob]))))
    if not values:
        raise RuntimeError(f"No usable bootstrap samples for K={k}.")
    return values


def stability_summary(X, ks=(2, 3, 4), threshold=0.75) -> pd.DataFrame:
    rows = []
    for k in ks:
        seed = seed_stability(X, k)
        boot = bootstrap_stability(X, k)
        rows.append({
            "K": k,
            "Seed ARI": seed,
            "Bootstrap ARI": float(np.mean(boot)),
            "Bootstrap ARI Std": float(np.std(boot, ddof=1)),
            "Bootstrap ARI Median": float(np.median(boot)),
            "Verdict": "Stable" if seed >= threshold and np.mean(boot) >= threshold else "Not stable",
        })
    return pd.DataFrame(rows)
