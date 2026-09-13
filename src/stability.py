from pathlib import Path
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "wholesale_customers.csv"
FIG_DIR = PROJECT_ROOT / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPEND_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
CATEGORICAL_COLS = ["Channel", "Region"]

RANDOM_STATE = 42
N_INIT = 20

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample

sns.set_style("whitegrid")
CANDIDATE_KS = [2, 3, 4]
STABILITY_SEEDS = [42, 7, 123, 2024, 99]
N_BOOTSTRAP = 50
STABILITY_THRESHOLD = 0.75
STABILITY_THRESHOLD_DESCRIPTION = (
    "A project-defined ARI threshold of 0.75 was used as a practical stability heuristic."
)

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    scaled_df = pd.read_csv(PROCESSED_DIR / "scaled_features.csv")
    df_raw = pd.read_csv(DATA_PATH)
    print("=" * 70)
    print("SECTION 1: LOAD SCALED FEATURES & RAW DATA")
    print("=" * 70)
    print(f"Scaled feature matrix shape: {scaled_df.shape}")
    print(f"df_raw shape: {df_raw.shape}")
    assert len(scaled_df) == len(df_raw), "Row count mismatch between scaled_features.csv and raw data."
    print("[OK] Row counts match and are index-aligned.")
    return df_raw, scaled_df.values

def train_final_models(X: np.ndarray) -> dict:
    """Train the final K-Means model for each candidate K, with a fixed
    random_state and n_init for full reproducibility."""
    print("\n" + "=" * 70)
    print("SECTION 2: TRAIN FINAL K-MEANS MODEL(S)")
    print("=" * 70)
    print(f"Fixed settings: random_state={RANDOM_STATE}, n_init={N_INIT!r}")

    models = {}
    for k in CANDIDATE_KS:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X)
        models[k] = {"model": km, "labels": labels}
        sizes = pd.Series(labels).value_counts().sort_index()
        print(f"\nK={k}: trained. Cluster sizes: {sizes.to_dict()} "
              f"(inertia={km.inertia_:.2f})")

    print("\n[OK] Final model(s) trained with locked random_state/n_init — "
          "re-running this script reproduces identical cluster assignments.")
    return models

def summarize_ari(values: pd.Series | np.ndarray, threshold: float = STABILITY_THRESHOLD) -> dict:
    """Return the complete ARI distribution summary required for stability reporting."""
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        raise ValueError("Cannot summarize an empty ARI series.")

    return {
        "mean_ari": float(series.mean()),
        "median_ari": float(series.median()),
        "std_ari": float(series.std(ddof=1)),
        "min_ari": float(series.min()),
        "max_ari": float(series.max()),
        "p25_ari": float(series.quantile(0.25)),
        "p75_ari": float(series.quantile(0.75)),
        "pass_rate": float((series >= threshold).mean()),
        "n_runs": int(series.size),
        "threshold": float(threshold),
    }


def check_stability_seeds(X: np.ndarray, k: int) -> pd.DataFrame:
    """Re-run K-Means with several seeds and compute pairwise ARI."""
    print(f"\n--- Seed-variation stability check for K={k} ---")

    seed_labels = {}
    for seed in STABILITY_SEEDS:
        km = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT)
        seed_labels[seed] = km.fit_predict(X)

    rows = []
    seeds = list(seed_labels.keys())
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari = adjusted_rand_score(seed_labels[seeds[i]], seed_labels[seeds[j]])
            rows.append({"seed_a": seeds[i], "seed_b": seeds[j], "ari": ari})

    ari_df = pd.DataFrame(rows)
    summary = summarize_ari(ari_df["ari"])
    print(ari_df.round(4).to_string(index=False))
    print(
        "Seed-variation ARI summary: "
        f"mean={summary['mean_ari']:.4f}, median={summary['median_ari']:.4f}, "
        f"std={summary['std_ari']:.4f}, min={summary['min_ari']:.4f}, "
        f"max={summary['max_ari']:.4f}, p25={summary['p25_ari']:.4f}, "
        f"p75={summary['p75_ari']:.4f}, pass rate={summary['pass_rate']:.1%}"
    )
    return ari_df


def check_stability_bootstrap(X: np.ndarray, k: int) -> pd.DataFrame:
    """Re-run K-Means on bootstrap resamples and compare OOB rows to baseline."""
    print(f"\n--- Bootstrap stability check for K={k} ({N_BOOTSTRAP} resamples) ---")

    baseline_km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    baseline_labels_full = baseline_km.fit_predict(X)

    n = X.shape[0]
    rows = []
    for b in range(N_BOOTSTRAP):
        boot_idx = resample(np.arange(n), replace=True, n_samples=n, random_state=b)
        oob_idx = np.setdiff1d(np.arange(n), np.unique(boot_idx))
        if len(oob_idx) < k * 2:
            continue

        boot_km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        boot_km.fit(X[boot_idx])

        boot_pred_oob = boot_km.predict(X[oob_idx])
        baseline_pred_oob = baseline_labels_full[oob_idx]
        ari = adjusted_rand_score(baseline_pred_oob, boot_pred_oob)
        rows.append({"bootstrap_run": b, "n_oob": len(oob_idx), "ari_vs_baseline": ari})

    boot_df = pd.DataFrame(rows)
    summary = summarize_ari(boot_df["ari_vs_baseline"])
    print(boot_df.round(4).to_string(index=False))
    print(
        "Bootstrap ARI summary: "
        f"mean={summary['mean_ari']:.4f}, median={summary['median_ari']:.4f}, "
        f"std={summary['std_ari']:.4f}, min={summary['min_ari']:.4f}, "
        f"max={summary['max_ari']:.4f}, p25={summary['p25_ari']:.4f}, "
        f"p75={summary['p75_ari']:.4f}, pass rate={summary['pass_rate']:.1%}"
    )
    return boot_df


def compute_stability_results(X: np.ndarray) -> dict:
    """Compute all candidate-K stability checks from one canonical routine."""
    results = {}
    for k in CANDIDATE_KS:
        results[k] = {
            "seed_ari": check_stability_seeds(X, k),
            "boot_ari": check_stability_bootstrap(X, k),
        }
    return results


def build_stability_summary_table(stability_results: dict) -> pd.DataFrame:
    """Create one rigorous long-form table containing all required ARI statistics."""
    rows = []
    for k, res in stability_results.items():
        for validation_name, values in [
            ("Seed-variation", res["seed_ari"]["ari"]),
            ("Bootstrap", res["boot_ari"]["ari_vs_baseline"]),
        ]:
            summary = summarize_ari(values)
            rows.append(
                {
                    "K": int(k),
                    "Validation": validation_name,
                    "Mean ARI": summary["mean_ari"],
                    "Median ARI": summary["median_ari"],
                    "Std ARI": summary["std_ari"],
                    "Minimum ARI": summary["min_ari"],
                    "Maximum ARI": summary["max_ari"],
                    "25th Percentile": summary["p25_ari"],
                    "75th Percentile": summary["p75_ari"],
                    "% Runs Above Threshold": summary["pass_rate"],
                    "Threshold": summary["threshold"],
                    "Runs": summary["n_runs"],
                }
            )
    return pd.DataFrame(rows)


def plot_stability_summary(stability_results: dict) -> None:
    """Bar chart comparing mean ARI across candidate K values."""
    print("\n" + "=" * 70)
    print("SECTION 6: VISUALIZE STABILITY RESULTS")
    print("=" * 70)

    ks = list(stability_results.keys())
    seed_means = [summarize_ari(stability_results[k]["seed_ari"]["ari"])["mean_ari"] for k in ks]
    boot_means = [summarize_ari(stability_results[k]["boot_ari"]["ari_vs_baseline"])["mean_ari"] for k in ks]

    x = np.arange(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, seed_means, width, label="Seed-variation ARI", color="steelblue")
    ax.bar(x + width / 2, boot_means, width, label="Bootstrap ARI", color="darkorange")
    ax.axhline(STABILITY_THRESHOLD, color="gray", linestyle="--", linewidth=1, label="Project-defined threshold (0.75)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Mean Adjusted Rand Index")
    ax.set_title("Cluster Stability: Mean ARI by K (Seed-Variation vs. Bootstrap)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/14_stability.png")


def document_stability_conclusion(stability_results: dict) -> str:
    """Summarize candidate stability using the complete ARI distribution statistics."""
    print("\n" + "=" * 70)
    print("SECTION 7: DOCUMENT STABILITY CONCLUSION")
    print("=" * 70)

    lines = [
        STABILITY_THRESHOLD_DESCRIPTION,
        "A candidate is classified as stable only when both seed-variation and bootstrap mean ARI meet this threshold.",
        "",
    ]

    for k, res in stability_results.items():
        seed = summarize_ari(res["seed_ari"]["ari"])
        boot = summarize_ari(res["boot_ari"]["ari_vs_baseline"])
        stable = (
            seed["mean_ari"] >= STABILITY_THRESHOLD
            and boot["mean_ari"] >= STABILITY_THRESHOLD
        )
        verdict = "STABLE" if stable else "NOT STABLE"
        lines.append(
            f"K={k}: {verdict}\n"
            f"  Seed-variation: mean={seed['mean_ari']:.4f}, median={seed['median_ari']:.4f}, "
            f"std={seed['std_ari']:.4f}, min={seed['min_ari']:.4f}, max={seed['max_ari']:.4f}, "
            f"p25={seed['p25_ari']:.4f}, p75={seed['p75_ari']:.4f}, "
            f"pass rate={seed['pass_rate']:.1%}\n"
            f"  Bootstrap: mean={boot['mean_ari']:.4f}, median={boot['median_ari']:.4f}, "
            f"std={boot['std_ari']:.4f}, min={boot['min_ari']:.4f}, max={boot['max_ari']:.4f}, "
            f"p25={boot['p25_ari']:.4f}, p75={boot['p75_ari']:.4f}, "
            f"pass rate={boot['pass_rate']:.1%}"
        )

    conclusion = "\n".join(lines)
    print(conclusion)
    print(
        "\n[NOTE] Stability is interpreted as a reproducibility check on the observed "
        "cluster assignments, not as proof that the clustering is the unique or true population structure."
    )
    return conclusion


def assign_labels_to_raw(df_raw: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Attach candidate K-Means labels to an untouched copy of the raw data."""
    df_labeled = df_raw.copy()
    for k, result in models.items():
        col_name = f"cluster_k{k}"
        df_labeled[col_name] = result["labels"]
        print(f"Added column '{col_name}' — value counts: {df_labeled[col_name].value_counts().sort_index().to_dict()}")
    return df_labeled


def save_stability_summary(stability_results: dict) -> None:
    summary_df = build_stability_summary_table(stability_results)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(PROCESSED_DIR / "stability_summary.csv", index=False)
    print("[OK] Full stability summary saved to stability_summary.csv")


def save_labeled_data(df_labeled: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(PROCESSED_DIR / "labeled_customers.csv", index=False)
    print(f"[OK] Saved labeled_customers.csv (shape: {df_labeled.shape})")
