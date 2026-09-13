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
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")
CHANNEL_MAP = {1: "Horeca (1)", 2: "Retail (2)"}

def cluster_column(selected_k: int) -> str:
    return f"cluster_k{int(selected_k)}"


def load_labeled_data(selected_k: int) -> pd.DataFrame:
    """Load raw data with candidate K-Means labels attached."""
    df = pd.read_csv(PROCESSED_DIR / "labeled_customers.csv")
    primary_col = cluster_column(selected_k)
    print("=" * 70)
    print("SECTION 1: LOAD LABELED DATA")
    print("=" * 70)
    print(f"Loaded labeled_customers.csv — shape: {df.shape}")
    print(f"Primary clustering for this phase: '{primary_col}'.")
    return df


def report_cluster_sizes(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Report cluster sizes as counts and percentage of the total 440 customers."""
    print("\n" + "=" * 70)
    print(f"SECTION 2: CLUSTER SIZES ({cluster_col})")
    print("=" * 70)

    sizes = df[cluster_col].value_counts().sort_index()
    pct = (sizes / len(df) * 100).round(1)
    size_table = pd.DataFrame({"count": sizes, "pct_of_440": pct})
    print(size_table.to_string())
    return size_table

def build_cluster_profiles(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Compute mean and median spend per category, per cluster."""
    print("\n" + "=" * 70)
    print(f"SECTION 3: PER-CLUSTER PROFILES ({cluster_col})")
    print("=" * 70)

    mean_profile = df.groupby(cluster_col)[SPEND_COLS].mean().round(1)
    median_profile = df.groupby(cluster_col)[SPEND_COLS].median().round(1)

    print("Mean spend per category, per cluster:")
    print(mean_profile.to_string())
    print("\nMedian spend per category, per cluster:")
    print(median_profile.to_string())

    return mean_profile

def plot_cluster_profile_bars(mean_profile: pd.DataFrame, cluster_col: str) -> None:
    """Grouped bar chart of mean spend per category, per cluster."""
    profile_pct = mean_profile.T  # categories as rows, clusters as columns
    fig, ax = plt.subplots(figsize=(11, 6))
    profile_pct.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title(f"Mean Spend per Category by Cluster ({cluster_col})")
    ax.set_ylabel("Mean Annual Spend")
    ax.set_xlabel("Product Category")
    ax.legend(title="Cluster")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fname = f"{FIG_DIR}/15_cluster_profile.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {fname}")


def rank_discriminating_features(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Rank features by descriptive separation strength using ANOVA F and eta-squared.

    The clusters were created from these same variables, so p-values are not treated
    as independent confirmatory evidence.
    """
    print("\n" + "=" * 70)
    print(f"SECTION 4: DESCRIPTIVE FEATURE SEPARATION ({cluster_col})")
    print("=" * 70)
    rows = []
    cluster_ids = sorted(df[cluster_col].unique())
    for col in SPEND_COLS:
        groups = [df.loc[df[cluster_col] == cid, col].values for cid in cluster_ids]
        f_stat, p_val = f_oneway(*groups)
        grand_mean = df[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df[col] - grand_mean) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total else 0.0
        rows.append({"feature": col, "F_statistic": f_stat, "eta_squared": eta2, "p_value": p_val})
    ranking = pd.DataFrame(rows).sort_values("F_statistic", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1
    print(ranking.round(4).to_string())
    print("\n[NOTE] F-statistic and eta-squared are descriptive because the clusters were constructed from these variables.")
    return ranking

def plot_feature_ranking(ranking: pd.DataFrame, cluster_col: str) -> None:
    """Bar chart of F-statistics per feature, ordered by discriminating power."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ranking["feature"][::-1], ranking["F_statistic"][::-1], color="teal")
    ax.set_xlabel("Descriptive ANOVA F-statistic")
    ax.set_title(f"Observed Feature Separation by Cluster ({cluster_col})")
    fig.tight_layout()
    fname = f"{FIG_DIR}/16_feature_discrimination.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {fname}")

def compare_high_low_spend_clusters(mean_profile: pd.DataFrame, cluster_col: str) -> dict:
    """Identify which cluster is the 'high total spend' cluster vs. 'low
    total spend' cluster, and which categories drive the separation between
    them."""
    print("\n" + "=" * 70)
    print(f"SECTION 5: HIGH-SPEND VS. LOW-SPEND CLUSTER COMPARISON ({cluster_col})")
    print("=" * 70)

    total_spend = mean_profile.sum(axis=1).sort_values(ascending=False)
    print(f"Total mean spend per cluster (sum across all 6 categories):")
    print(total_spend.round(1).to_string())

    high_cluster = total_spend.index[0]
    low_cluster = total_spend.index[-1]
    print(f"\nHighest total-spend cluster: {high_cluster} "
          f"(total mean spend = {total_spend.iloc[0]:.1f})")
    print(f"Lowest total-spend cluster: {low_cluster} "
          f"(total mean spend = {total_spend.iloc[-1]:.1f})")

    diff = (mean_profile.loc[high_cluster] - mean_profile.loc[low_cluster]).sort_values(ascending=False)
    print(f"\nCategory-level difference (high-spend cluster minus low-spend cluster), "
          "largest gap first:")
    print(diff.round(1).to_string())
    print(f"\n[FINDING] The category driving the largest separation between the "
          f"high- and low-spend clusters is '{diff.index[0]}' "
          f"(gap = {diff.iloc[0]:.1f}).")

    return {"high_cluster": high_cluster, "low_cluster": low_cluster, "category_gaps": diff}

def crosstab_channel_region(df: pd.DataFrame, cluster_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-tabulate cluster membership against Channel and Region (both
    held out of the clustering feature set in Phase 2) as a sanity check on
    whether clusters align with the known Horeca/Retail channel split."""
    print("\n" + "=" * 70)
    print(f"SECTION 6: CROSS-TABULATE CLUSTERS VS. CHANNEL & REGION ({cluster_col})")
    print("=" * 70)

    channel_map = {1: "Horeca (1)", 2: "Retail (2)"}
    channel_ct = pd.crosstab(df[cluster_col], df["Channel"].map(channel_map))
    channel_ct_pct = (channel_ct.div(channel_ct.sum(axis=1), axis=0) * 100).round(1)
    print("Cluster x Channel (counts):")
    print(channel_ct.to_string())
    print("\nCluster x Channel (row %):")
    print(channel_ct_pct.to_string())

    region_ct = pd.crosstab(df[cluster_col], df["Region"])
    region_ct_pct = (region_ct.div(region_ct.sum(axis=1), axis=0) * 100).round(1)
    print("\nCluster x Region (counts):")
    print(region_ct.to_string())
    print("\nCluster x Region (row %):")
    print(region_ct_pct.to_string())

    # Quantify channel alignment: for each cluster, what % belongs to its
    # majority channel? High values (>>50%) suggest clusters mirror Channel.
    majority_pct = channel_ct_pct.max(axis=1)
    print(f"\n[FINDING] Majority-channel share per cluster: {majority_pct.round(1).to_dict()}")
    if (majority_pct > 70).all():
        print("[FINDING] Clusters show strong alignment with the existing Channel "
              "variable (each cluster is dominated by one channel) — clustering "
              "on spend alone largely rediscovers the known Horeca/Retail split.")
    else:
        print("[FINDING] Clusters do NOT strongly mirror Channel — spend-based "
              "segmentation reveals structure beyond the existing channel label.")

    return channel_ct, region_ct

def plot_channel_crosstab(channel_ct: pd.DataFrame, cluster_col: str) -> None:
    """Stacked bar chart of Channel composition per cluster."""
    fig, ax = plt.subplots(figsize=(7, 5))
    channel_ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title(f"Channel Composition per Cluster ({cluster_col})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.legend(title="Channel")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fname = f"{FIG_DIR}/17_channel_crosstab.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {fname}")

def plot_region_crosstab(region_ct: pd.DataFrame, cluster_col: str) -> None:
    """Stacked bar chart of Region composition per cluster."""
    fig, ax = plt.subplots(figsize=(8, 5))
    region_ct.plot(kind="bar", stacked=True, ax=ax, colormap="Pastel1")
    ax.set_title(f"Region Composition per Cluster ({cluster_col})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.legend(title="Region")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fname = f"{FIG_DIR}/18_region_crosstab.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {fname}")

def brief_secondary_k3_summary(df: pd.DataFrame, selected_k: int) -> None:
    """Brief profile of K=3 only when it is not the selected solution."""
    print("\n" + "=" * 70)
    print("SECTION 7: BRIEF SECONDARY COMPARISON — K=3 (EXPLORATORY)")
    print("=" * 70)
    secondary_k = 3 if int(selected_k) != 3 and "cluster_k3" in df.columns else None
    if secondary_k is None:
        print("[NOTE] No separate K=3 secondary comparison is shown because K=3 is the selected solution or is unavailable.")
        return
    secondary_col = cluster_column(secondary_k)

    print("[NOTE] K=3 is retained only as a secondary reference when it is not the selected solution.")

    sizes = df[secondary_col].value_counts().sort_index()
    pct = (sizes / len(df) * 100).round(1)
    print(f"\nK=3 cluster sizes: {sizes.to_dict()} (% of 440: {pct.to_dict()})")

    mean_profile_k3 = df.groupby(secondary_col)[SPEND_COLS].mean().round(1)
    print("\nK=3 mean spend per category, per cluster:")
    print(mean_profile_k3.to_string())

def load_pca_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    scaled_df = pd.read_csv(PROCESSED_DIR / "scaled_features.csv")
    labeled_df = pd.read_csv(PROCESSED_DIR / "labeled_customers.csv")
    assert len(scaled_df) == len(labeled_df), "Row count mismatch between scaled_features.csv and labeled_customers.csv."
    return scaled_df, labeled_df

def apply_pca(scaled_df: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    """Reduce the 6 scaled spend features to 2 principal components for
    2D visualization. Not used for clustering itself (Phase 6 clustered on
    the full 6-feature space) — PCA here serves the visualization need
    flagged in Phase 3's multicollinearity decision (Option A)."""
    print("\n" + "=" * 70)
    print("SECTION 2: APPLY PCA (2 COMPONENTS)")
    print("=" * 70)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(scaled_df[SPEND_COLS])

    explained_var = pca.explained_variance_ratio_
    print(f"PC1 explains {explained_var[0]*100:.1f}% of variance")
    print(f"PC2 explains {explained_var[1]*100:.1f}% of variance")
    print(f"Combined (PC1+PC2): {explained_var.sum()*100:.1f}% of total variance")

    loadings = pd.DataFrame(
        pca.components_.T, columns=["PC1", "PC2"], index=SPEND_COLS
    )
    print("\nPCA component loadings (feature contribution to each PC):")
    print(loadings.round(3).to_string())

    top_pc1 = loadings["PC1"].abs().idxmax()
    top_pc2 = loadings["PC2"].abs().idxmax()
    print(f"\n[FINDING] PC1 is most driven by '{top_pc1}' (consistent with Phase 7's "
          f"ANOVA finding that Grocery/Detergents_Paper/Milk dominate cluster "
          "separation). PC2 is most driven by "
          f"'{top_pc2}'.")

    return pcs, pca

def plot_pca_loadings(pca: PCA) -> None:
    """Bar chart of each feature's loading on PC1 and PC2, to make the
    PCA finding in Section 2 visually verifiable rather than table-only."""
    print("\n" + "=" * 70)
    print("SECTION 3: PCA LOADINGS VISUALIZATION")
    print("=" * 70)

    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=SPEND_COLS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    loadings["PC1"].sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Feature Loadings on PC1")
    axes[0].set_xlabel("Loading")

    loadings["PC2"].sort_values().plot(kind="barh", ax=axes[1], color="darkorange")
    axes[1].set_title("Feature Loadings on PC2")
    axes[1].set_xlabel("Loading")

    fig.suptitle("PCA Component Loadings by Feature", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "19_pca_loadings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/19_pca_loadings.png")

def plot_pca_cluster_scatter(
    pcs: np.ndarray, labeled_df: pd.DataFrame, pca: PCA, cluster_col: str
) -> None:
    """Primary visualization: PC1 vs. PC2 scatter, colored by cluster label,
    with axis labels annotating the variance explained by each component."""
    print("\n" + "=" * 70)
    print("SECTION 4: PCA SCATTER PLOT COLORED BY CLUSTER")
    print("=" * 70)

    explained_var = pca.explained_variance_ratio_
    plot_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Cluster": labeled_df[cluster_col].astype(str),
    })

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster",
                     palette="Set2", s=60, alpha=0.75, ax=ax)
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance explained)")
    selected_k = int(cluster_col.replace("cluster_k", ""))
    ax.set_title(f"K-Means Clusters (K={selected_k}) Visualized in PCA Space\n"
                 f"(PC1+PC2 capture {explained_var.sum()*100:.1f}% of total variance)")
    ax.legend(title="Cluster")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "20_pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/20_pca_clusters.png")
    print(f"[NOTE] Only {explained_var.sum()*100:.1f}% of total variance is captured "
          "in this 2D view — cluster separation may look less clean here than in "
          "the full 6-feature space K-Means actually used (Phase 6). This plot is "
          "for visualization only; it is not the basis the clustering itself relied on.")

def plot_pca_channel_overlay(
    pcs: np.ndarray, labeled_df: pd.DataFrame, pca: PCA, cluster_col: str
) -> None:
    """Overlay Channel as marker shape on the same PCA scatter (color = cluster,
    marker shape = Channel) to visually compare cluster boundaries against the
    known Horeca/Retail split, following up on Phase 7's cross-tab finding."""
    print("\n" + "=" * 70)
    print("SECTION 5: PCA SCATTER OVERLAID WITH CHANNEL MARKERS")
    print("=" * 70)

    explained_var = pca.explained_variance_ratio_
    plot_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Cluster": labeled_df[cluster_col].astype(str),
        "Channel": labeled_df["Channel"].map(CHANNEL_MAP),
    })

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster", style="Channel",
                     palette="Set2", s=70, alpha=0.75, ax=ax)
    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance explained)")
    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance explained)")
    ax.set_title("Clusters (color) vs. Channel (marker shape) in PCA Space")
    ax.legend(title="Cluster / Channel", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "21_pca_channel_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/21_pca_channel_overlay.png")
    print("[NOTE] This illustrates the post-hoc relationship between the "
          "canonical cluster labels and Channel; mixed cases remain visible "
          "where the minority channel appears within a cluster.")



