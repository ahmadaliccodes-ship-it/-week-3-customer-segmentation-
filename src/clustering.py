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
import matplotlib.cm as cm
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_style("whitegrid")
K_RANGE = range(2, 11)
LINKAGE_METHOD = "ward"

def load_scaled_features() -> pd.DataFrame:
    scaled_df = pd.read_csv(PROCESSED_DIR / "scaled_features.csv")
    print("=" * 70)
    print("SECTION 1: LOAD TRANSFORMED & SCALED FEATURE MATRIX")
    print("=" * 70)
    print(f"Loaded scaled_features.csv — shape: {scaled_df.shape}")
    print(f"Columns: {list(scaled_df.columns)}")
    print(f"Mean per column (expect ~0): {scaled_df.mean().round(3).to_dict()}")
    print(f"Std per column (expect ~1): {scaled_df.std().round(3).to_dict()}")
    return scaled_df

def fit_kmeans_range(X: np.ndarray) -> pd.DataFrame:
    """Fit K-Means for each K in K_RANGE, recording inertia (for the elbow
    method) and average silhouette score (for silhouette analysis)."""
    print("\n" + "=" * 70)
    print("SECTION 2: K-MEANS — FIT ACROSS K = 2 TO 10")
    print("=" * 70)

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  K={k:2d}  inertia={km.inertia_:10.2f}  silhouette={sil:.4f}")

    kmeans_results = pd.DataFrame(rows)
    print("\n[OK] K-Means fitted for all K in range with fixed random_state="
          f"{RANDOM_STATE}, n_init={N_INIT} for reproducibility.")
    return kmeans_results

def plot_elbow(kmeans_results: pd.DataFrame) -> None:
    """Plot inertia vs. K to visually identify the 'elbow' point."""
    print("\n" + "=" * 70)
    print("SECTION 3: K-MEANS — ELBOW METHOD PLOT")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kmeans_results["k"], kmeans_results["inertia"], marker="o", color="steelblue")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_title("Elbow Method: Inertia vs. K (K-Means)")
    ax.set_xticks(list(K_RANGE))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_elbow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/09_elbow.png")

    print(
        "[NOTE] The inertia curve does not show a uniquely sharp elbow. "
        "The elbow plot is therefore treated as supporting evidence only; "
        "silhouette score, clustering stability, hierarchical clustering comparison, "
        "and business interpretability are given greater weight when selecting K."
    )

def fit_hierarchical_range(X: np.ndarray) -> pd.DataFrame:
    """Fit Agglomerative Hierarchical Clustering (Ward linkage) for each K
    in K_RANGE, recording average silhouette score."""
    print("\n" + "=" * 70)
    print("SECTION 4: HIERARCHICAL CLUSTERING — FIT ACROSS K = 2 TO 10")
    print("=" * 70)
    print(f"Linkage method: {LINKAGE_METHOD} (assumption — not specified in plan)")

    rows = []
    for k in K_RANGE:
        agg = AgglomerativeClustering(n_clusters=k, linkage=LINKAGE_METHOD)
        labels = agg.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": sil})
        print(f"  K={k:2d}  silhouette={sil:.4f}")

    hier_results = pd.DataFrame(rows)
    print("\n[NOTE] Hierarchical clustering has no 'inertia' concept, so the "
          "elbow method (Step 13) applies to K-Means only, as specified in "
          "the plan. Silhouette score is the comparison metric for both.")
    return hier_results

def plot_dendrogram(X: np.ndarray) -> None:
    """Plot the Ward-linkage dendrogram to visually cross-check how many
    natural groupings the hierarchy suggests, independent of a chosen K."""
    print("\n" + "=" * 70)
    print("SECTION 5: HIERARCHICAL CLUSTERING — DENDROGRAM")
    print("=" * 70)

    Z = linkage(X, method=LINKAGE_METHOD)

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
               leaf_rotation=90, leaf_font_size=8, show_contracted=True)
    ax.set_title(f"Dendrogram ({LINKAGE_METHOD.title()} Linkage, last 30 merges shown)")
    ax.set_xlabel("Customer clusters (or individual customers)")
    ax.set_ylabel("Distance (Ward)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/10_dendrogram.png")
    print("[NOTE] Dendrogram is a visual cross-check on cluster structure, not "
          "itself a numeric criterion for choosing K — used alongside silhouette.")

def plot_silhouette_comparison(kmeans_results: pd.DataFrame, hier_results: pd.DataFrame) -> None:
    """Plot average silhouette score vs. K for both algorithms on one chart —
    the primary decision-making figure per the plan ('more decisive than
    elbow given the skewed data')."""
    print("\n" + "=" * 70)
    print("SECTION 6: SILHOUETTE ANALYSIS — COMPARISON PLOT")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(kmeans_results["k"], kmeans_results["silhouette"], marker="o",
            label="K-Means", color="steelblue")
    ax.plot(hier_results["k"], hier_results["silhouette"], marker="s",
            label="Hierarchical (Ward)", color="darkorange")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("Silhouette Score vs. K: K-Means vs. Hierarchical Clustering")
    ax.set_xticks(list(K_RANGE))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_silhouette_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/11_silhouette_comparison.png")

    combined = kmeans_results[["k", "silhouette"]].merge(
        hier_results[["k", "silhouette"]], on="k", suffixes=("_kmeans", "_hierarchical"))
    print("\nSilhouette scores side by side:")
    print(combined.round(4).to_string(index=False))

def plot_detailed_silhouette(X: np.ndarray, k: int) -> float:
    """Per-sample silhouette plot for a specific K (K-Means), showing cluster
    sizes and whether any cluster's samples fall below the average score —
    a more rigorous silhouette diagnostic than the average alone."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(X)
    sil_avg = silhouette_score(X, labels)
    sample_sil = silhouette_samples(X, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(k):
        cluster_sil = np.sort(sample_sil[labels == i])
        size = cluster_sil.shape[0]
        y_upper = y_lower + size
        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--", label=f"Average = {sil_avg:.3f}")
    ax.set_title(f"Detailed Silhouette Plot — K-Means, K={k}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"12_silhouette_k{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/12_silhouette_k{k}.png (avg silhouette={sil_avg:.4f})")
    return sil_avg

def run_detailed_silhouette_for_top_candidates(X: np.ndarray, kmeans_results: pd.DataFrame) -> None:
    """Generate detailed silhouette plots for the top 2 K-Means candidates
    by average silhouette score, to support the final K decision."""
    print("\n" + "=" * 70)
    print("SECTION 7: DETAILED SILHOUETTE PLOTS FOR TOP CANDIDATE K VALUES")
    print("=" * 70)

    top_candidates = kmeans_results.sort_values("silhouette", ascending=False).head(2)["k"].tolist()
    print(f"Top 2 K-Means candidates by average silhouette score: {top_candidates}")
    for k in top_candidates:
        plot_detailed_silhouette(X, k)

SELECTION_PATH = PROCESSED_DIR / "k_selection.json"


def _feature_fingerprint(X: np.ndarray) -> str:
    """Return a deterministic fingerprint for the feature matrix used for K selection."""
    array = np.asarray(X, dtype=float)
    return __import__("hashlib").sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def _load_saved_selection(X: np.ndarray | None) -> dict | None:
    if X is None or not SELECTION_PATH.exists():
        return None

    with open(SELECTION_PATH, "r", encoding="utf-8") as f:
        saved = json.load(f)

    if saved.get("feature_fingerprint") != _feature_fingerprint(X):
        return None
    return saved


def _write_selection(selection: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_PATH, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)


def select_k(
    X: np.ndarray | None = None,
    kmeans_results: pd.DataFrame | None = None,
    hier_results: pd.DataFrame | None = None,
    stability_results: dict | None = None,
    force_recompute: bool = False,
) -> dict:
    """Select one canonical K and persist the exact decision for all phases.

    Decision rule:
    1. Rank K-Means candidates by silhouette score.
    2. Treat a candidate as stable only when both seed-variation mean ARI and
       bootstrap mean ARI meet the project-defined 0.75 heuristic.
    3. Select the highest-silhouette stable candidate.
    4. If no candidate is stable, fall back to the highest-silhouette K and
       explicitly flag that condition.

    When a compatible saved decision exists, it is returned unchanged so
    Phases 5-10 and run_pipeline.py all use one identical K decision.
    """
    saved = None if force_recompute else _load_saved_selection(X)
    if saved is not None:
        print(
            f"[OK] Loaded canonical K selection from {SELECTION_PATH.name}: "
            f"K={saved['selected_k']}"
        )
        return saved

    if X is None:
        if not SELECTION_PATH.exists():
            raise FileNotFoundError(
                f"{SELECTION_PATH} does not exist. Run Phase 5 to create the canonical K decision."
            )
        with open(SELECTION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    if kmeans_results is None:
        kmeans_results = fit_kmeans_range(X)
    if hier_results is None:
        hier_results = fit_hierarchical_range(X)

    if stability_results is None:
        # Local import avoids a module-level clustering <-> stability cycle.
        from . import stability
        stability_results = stability.compute_stability_results(X)

    stable_rows = []
    for k in stability.CANDIDATE_KS:
        if k not in set(kmeans_results["k"].astype(int)):
            continue
        k_res = stability_results[k]
        seed_summary = stability.summarize_ari(k_res["seed_ari"]["ari"])
        boot_summary = stability.summarize_ari(k_res["boot_ari"]["ari_vs_baseline"])
        stable_rows.append(
            {
                "k": int(k),
                "kmeans_silhouette": float(
                    kmeans_results.loc[kmeans_results["k"] == k, "silhouette"].iloc[0]
                ),
                "seed_mean_ari": seed_summary["mean_ari"],
                "bootstrap_mean_ari": boot_summary["mean_ari"],
                "stable": bool(
                    seed_summary["mean_ari"] >= stability.STABILITY_THRESHOLD
                    and boot_summary["mean_ari"] >= stability.STABILITY_THRESHOLD
                ),
            }
        )

    stability_df = pd.DataFrame(stable_rows)
    if stability_df.empty:
        raise ValueError("No K candidates overlap between K-Means results and stability validation.")

    stable_candidates = stability_df[stability_df["stable"]]
    if not stable_candidates.empty:
        selected_row = stable_candidates.sort_values(
            ["kmeans_silhouette", "k"], ascending=[False, True]
        ).iloc[0]
        stability_rule = (
            "Selected the highest-silhouette K among candidates that pass both "
            "seed-variation and bootstrap stability checks."
        )
    else:
        selected_row = stability_df.sort_values(
            ["kmeans_silhouette", "k"], ascending=[False, True]
        ).iloc[0]
        stability_rule = (
            "No validated candidate met the project-defined stability heuristic; "
            "selected the highest-silhouette K and flagged the stability limitation."
        )

    best_hier_row = hier_results.loc[hier_results["silhouette"].idxmax()]
    selected_k = int(selected_row["k"])
    selected_stability = next(
        row for row in stable_rows if row["k"] == selected_k
    )

    reason = (
        f"K={selected_k} was selected because it has the highest K-Means silhouette "
        f"among the stability-qualified candidates (silhouette={selected_stability['kmeans_silhouette']:.4f}). "
        f"Seed-variation mean ARI={selected_stability['seed_mean_ari']:.4f}; "
        f"bootstrap mean ARI={selected_stability['bootstrap_mean_ari']:.4f}. "
        f"{stability_rule} "
        f"The hierarchical silhouette cross-check peaks at K={int(best_hier_row['k'])} "
        f"(silhouette={float(best_hier_row['silhouette']):.4f})."
    )

    selection = {
        "selected_k": selected_k,
        "reason": reason,
        "silhouette": selected_stability["kmeans_silhouette"],
        "stability": {
            "seed_mean_ari": selected_stability["seed_mean_ari"],
            "bootstrap_mean_ari": selected_stability["bootstrap_mean_ari"],
            "threshold": stability.STABILITY_THRESHOLD,
            "seed_pass": bool(
                selected_stability["seed_mean_ari"] >= stability.STABILITY_THRESHOLD
            ),
            "bootstrap_pass": bool(
                selected_stability["bootstrap_mean_ari"] >= stability.STABILITY_THRESHOLD
            ),
            "pass": bool(selected_stability["stable"]),
        },
        "hierarchical_best_k": int(best_hier_row["k"]),
        "hierarchical_best_silhouette": float(best_hier_row["silhouette"]),
        "stable_candidates": [int(k) for k in stable_candidates["k"].tolist()],
        "decision_rule": (
            "Highest K-Means silhouette among stability-qualified candidates; "
            "stability qualification requires seed-variation and bootstrap mean ARI "
            ">= 0.75. If none qualify, use highest silhouette with a stability warning."
        ),
        "candidate_stability": stable_rows,
        "feature_fingerprint": _feature_fingerprint(X),
    }
    _write_selection(selection)

    print("\n" + "=" * 70)
    print("CANONICAL K SELECTION")
    print("=" * 70)
    print(f"Selected K: {selected_k}")
    print(f"Silhouette: {selection['silhouette']:.4f}")
    print(
        f"Stability: seed mean ARI={selection['stability']['seed_mean_ari']:.4f}, "
        f"bootstrap mean ARI={selection['stability']['bootstrap_mean_ari']:.4f}, "
        f"threshold={selection['stability']['threshold']:.2f}"
    )
    print(f"Reason: {reason}")
    print(f"[OK] Saved canonical decision to {SELECTION_PATH}")

    return selection

