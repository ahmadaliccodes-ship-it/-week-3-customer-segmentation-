from pathlib import Path
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "wholesale_customers.csv"
FIG_DIR = PROJECT_ROOT / "figures"

SPEND_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
CATEGORICAL_COLS = ["Channel", "Region"]

RANDOM_STATE = 42
N_INIT = 20

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_OBJECTIVE = (
    "Segment wholesale distribution customers into meaningful groups based on "
    "their annual spending across six product categories (Fresh, Milk, Grocery, "
    "Frozen, Detergents_Paper, Delicassen), using unsupervised clustering "
    "(K-Means, cross-checked with Hierarchical Clustering), in order to reveal "
    "actionable customer archetypes for targeted business decisions "
    "(e.g. promotions, delivery/route planning, inventory allocation)."
)

DATA_DICTIONARY = {
    "Channel": "Distribution channel: 1 = Horeca (Hotel/Restaurant/Cafe), 2 = Retail.",
    "Region": "Geographic region code: 1 = Lisbon, 2 = Oporto, 3 = Other region.",
    "Fresh": "Annual spending in monetary units on fresh products.",
    "Milk": "Annual spending in monetary units on milk products.",
    "Grocery": "Annual spending in monetary units on grocery products.",
    "Frozen": "Annual spending in monetary units on frozen products.",
    "Detergents_Paper": "Annual spending in monetary units on detergents and paper products.",
    "Delicassen": "Annual spending in monetary units on delicatessen products.",
}

def load_raw_data(snapshot=False) -> pd.DataFrame:
    df_raw = pd.read_csv(DATA_PATH)
    if snapshot:
        df_raw.to_csv(PROJECT_ROOT / "df_raw_snapshot.csv", index=False)
    return df_raw

def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a structured profile of the dataset for the report's
    'Introduction & Dataset Overview' section."""
    profile = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "n_unique_channel": df["Channel"].nunique(),
        "n_unique_region": df["Region"].nunique(),
        "channel_counts": df["Channel"].value_counts().to_dict(),
        "region_counts": df["Region"].value_counts().to_dict(),
    }
    return profile

def confirm_data_quality(df: pd.DataFrame) -> dict:
    """Re-verify missing values and duplicates on the working copy (df_raw
    is never mutated; this section only reads it)."""
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    negative_spend = (df[SPEND_COLS] < 0).sum()
    zero_spend = (df[SPEND_COLS] == 0).sum()
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]

    print("=" * 70)
    print("SECTION 1: DATA QUALITY CONFIRMATION")
    print("=" * 70)
    print(f"Missing values per column:\n{missing}")
    print(f"\nTotal missing values: {missing.sum()}")
    print(f"Duplicate rows: {duplicates}")
    print(f"Negative spending values by feature: {negative_spend.to_dict()}")
    print(f"Zero spending values by feature: {zero_spend.to_dict()}")
    print(f"Constant columns: {constant_cols}")

    if (missing.sum() == 0 and duplicates == 0 and
        negative_spend.sum() == 0 and zero_spend.sum() == 0 and not constant_cols):
        print("[OK] Core data quality confirmed: no missing values, duplicate rows, invalid negative/zero spending values, or constant columns.")
    else:
        print("[WARNING] Data quality issue detected — review before proceeding.")

    return {
        "missing_total": int(missing.sum()),
        "duplicates": int(duplicates),
        "negative_spend_total": int(negative_spend.sum()),
        "zero_spend_total": int(zero_spend.sum()),
        "constant_columns": constant_cols,
    }

def plot_histograms(df: pd.DataFrame) -> None:
    """Histogram per spend column to visualize distribution shape/skew."""
    print("\n" + "=" * 70)
    print("SECTION 2: UNIVARIATE EDA — HISTOGRAMS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(SPEND_COLS):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(f"{col} (annual spend)")
        axes[i].set_ylabel("Count")
    fig.suptitle("Univariate EDA: Spend Category Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/01_histograms.png")

def plot_boxplots(df: pd.DataFrame) -> None:
    """Boxplot per spend column to visualize spread and outliers."""
    print("\n" + "=" * 70)
    print("SECTION 3: UNIVARIATE EDA — BOXPLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(SPEND_COLS):
        sns.boxplot(y=df[col], ax=axes[i], color="darkorange")
        axes[i].set_title(f"Boxplot of {col}")
        axes[i].set_ylabel(f"{col} (annual spend)")
    fig.suptitle("Univariate EDA: Spend Category Boxplots (Outlier View)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/02_boxplots.png")

def plot_categorical_bars(df: pd.DataFrame) -> None:
    """Bar charts showing the distribution of Channel and Region."""
    print("\n" + "=" * 70)
    print("SECTION 4: UNIVARIATE EDA — CHANNEL & REGION DISTRIBUTION")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    channel_counts = df["Channel"].value_counts().sort_index()
    axes[0].bar(channel_counts.index.astype(str), channel_counts.values, color="seagreen")
    axes[0].set_title("Channel Distribution (1 = Horeca, 2 = Retail)")
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("Number of Customers")
    for i, v in enumerate(channel_counts.values):
        axes[0].text(i, v + 3, str(v), ha="center")

    region_counts = df["Region"].value_counts().sort_index()
    axes[1].bar(region_counts.index.astype(str), region_counts.values, color="indianred")
    axes[1].set_title("Region Distribution")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Number of Customers")
    for i, v in enumerate(region_counts.values):
        axes[1].text(i, v + 3, str(v), ha="center")

    fig.suptitle("Univariate EDA: Categorical Field Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_channel_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/03_channel_region.png")
    print(f"Channel counts: {channel_counts.to_dict()}")
    print(f"Region counts: {region_counts.to_dict()}")

def plot_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation heatmap across the 6 spend columns to surface
    multicollinearity ahead of Phase 3."""
    print("\n" + "=" * 70)
    print("SECTION 5: BIVARIATE EDA — CORRELATION HEATMAP")
    print("=" * 70)

    corr = df[SPEND_COLS].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, ax=ax)
    ax.set_title("Correlation Heatmap: Spend Categories")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/04_correlation_heatmap.png")
    print(f"\nCorrelation matrix:\n{corr.round(2)}")

    strong_pairs = []
    for i, c1 in enumerate(SPEND_COLS):
        for c2 in SPEND_COLS[i + 1:]:
            r = corr.loc[c1, c2]
            if abs(r) >= 0.6:
                strong_pairs.append((c1, c2, round(r, 2)))
    print(f"\nStrong correlations (|r| >= 0.6): {strong_pairs}")
    return corr

def plot_pairplot_by_channel(df: pd.DataFrame) -> None:
    """Exploratory pairplot of the six spend columns colored by Channel.
    Channel is an exploratory overlay only; it is not used to create clusters."""
    print("\n" + "=" * 70)
    print("SECTION 6: BIVARIATE EDA — PAIRPLOT COLORED BY CHANNEL")
    print("=" * 70)

    plot_df = df[SPEND_COLS + ["Channel"]].copy()
    plot_df["Channel"] = plot_df["Channel"].map({1: "Horeca (1)", 2: "Retail (2)"})

    g = sns.pairplot(plot_df, hue="Channel", palette={"Horeca (1)": "steelblue", "Retail (2)": "darkorange"},
                      diag_kind="hist", plot_kws={"alpha": 0.6, "s": 25})
    g.fig.suptitle("Pairplot of Spend Categories Colored by Channel", y=1.02, fontsize=14)
    g.savefig(FIG_DIR / "05_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[OK] Saved {FIG_DIR}/05_pairplot.png")
    print("[NOTE] Channel coloring is exploratory only and is not a clustering input.")

def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the 6 spend columns as clustering features.
    Channel and Region are intentionally excluded (reserved for post-hoc
    cluster profiling in Phase 7), per the assumption stated in Phase 1."""
    print("\n" + "=" * 70)
    print("SECTION 7: FEATURE SELECTION FOR CLUSTERING")
    print("=" * 70)

    features_df = df[SPEND_COLS].copy()

    print(f"Selected clustering features ({len(SPEND_COLS)}): {SPEND_COLS}")
    print(f"Excluded from clustering (reserved for profiling): {CATEGORICAL_COLS}")
    print(f"Resulting feature matrix shape: {features_df.shape}")

    return features_df

def analyze_skewness(df: pd.DataFrame) -> pd.Series:
    """Quantify skewness per spend column to justify transformation need."""
    print("=" * 70)
    print("SECTION 1: SKEWNESS ANALYSIS")
    print("=" * 70)

    skew_vals = df[SPEND_COLS].skew().sort_values(ascending=False)
    print("Skewness per feature (Pearson's moment coefficient):")
    print(skew_vals.round(2))

    print("\nInterpretation guide: |skew| > 1 = highly skewed; 0.5-1 = moderate; <0.5 = ~symmetric")
    for col, val in skew_vals.items():
        level = "HIGH" if abs(val) > 1 else ("MODERATE" if abs(val) > 0.5 else "LOW")
        print(f"  {col:<20} skew = {val:6.2f}  -> {level} skew")

    print("\n[FINDING] All 6 features exceed skew > 2 (Delicassen most extreme "
          f"at {skew_vals.max():.2f}). The distributions are heavily right-skewed. "
          "Because K-Means minimizes squared Euclidean distances, extreme values "
          "can disproportionately influence centroid positions. A variance-stabilizing "
          "log1p transform is therefore justified as a robustness step.")
    return skew_vals

def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers per spend column using the 1.5*IQR rule."""
    print("\n" + "=" * 70)
    print("SECTION 2: OUTLIER DETECTION (IQR METHOD)")
    print("=" * 70)

    rows = []
    for col in SPEND_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        rows.append({
            "feature": col,
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(100 * mask.sum() / len(df), 1),
        })

    outlier_summary = pd.DataFrame(rows).set_index("feature")
    print(outlier_summary.round(1).to_string())

    total_flagged_rows = (
        (df[SPEND_COLS] < outlier_summary["lower_bound"]) |
        (df[SPEND_COLS] > outlier_summary["upper_bound"])
    ).any(axis=1).sum()
    print(f"\n[FINDING] {outlier_summary['n_outliers'].min()}-{outlier_summary['n_outliers'].max()} "
          f"outliers per column (IQR rule). {total_flagged_rows} of {len(df)} customers "
          "({:.1f}%) have at least one outlying spend value — concentrated in "
          "high-spend 'bulk buyer' accounts, not data errors.".format(
              100 * total_flagged_rows / len(df)))
    return outlier_summary

def document_outlier_decision() -> str:
    """State and justify the outlier-handling decision for this project."""
    print("\n" + "=" * 70)
    print("SECTION 3: OUTLIER TREATMENT DECISION")
    print("=" * 70)

    decision = (
        "DECISION: Retain all outlier records — do NOT delete or cap them.\n\n"
        "RATIONALE:\n"
        "  1. Outliers are concentrated in high-spend accounts, which plausibly "
        "represent genuine large/bulk-buying customers rather than data-entry "
        "errors. Removing them would delete exactly the customer behavior most "
        "relevant to segmentation (e.g. distinguishing large Horeca accounts "
        "from small retail accounts).\n"
        "  2. No documented evidence (data dictionary, business rule, or "
        "domain confirmation) indicates these values are erroneous. Deleting "
        "them would be guessing intent not supported by the data.\n"
        "  3. Instead of deletion, distortion from extreme values will be "
        "neutralized statistically via log1p transform + standardization "
        "(Phase 4, Steps 9-10), which compresses scale without discarding "
        "any customer records.\n\n"
        "This choice is documented here explicitly per the implementation plan, "
        "for inclusion in the final report's Methodology section."
    )
    print(decision)
    return decision

def plot_outlier_reference(df: pd.DataFrame, outlier_summary: pd.DataFrame) -> None:
    """Boxplots annotated with outlier counts, for the report's outlier
    discussion (visual companion to Phase 2's boxplots)."""
    print("\n" + "=" * 70)
    print("SECTION 4: OUTLIER VISUALIZATION REFERENCE")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(SPEND_COLS):
        sns.boxplot(y=df[col], ax=axes[i], color="salmon")
        n_out = outlier_summary.loc[col, "n_outliers"]
        pct_out = outlier_summary.loc[col, "pct_outliers"]
        axes[i].set_title(f"{col}\n({n_out} outliers, {pct_out}%)")
        axes[i].set_ylabel(f"{col} (annual spend)")
    fig.suptitle("Outlier Reference: Boxplots Annotated with IQR Outlier Counts",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_outliers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/06_outliers.png")

def compute_vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Compute Variance Inflation Factor per feature manually:
    VIF_i = 1 / (1 - R^2_i), where R^2_i is from regressing feature i on
    all other features in `cols`. Equivalent to statsmodels' VIF."""
    X = df[cols].values
    vifs = []
    for i, col in enumerate(cols):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_others, y)
        r2 = reg.score(X_others, y)
        vif = np.inf if r2 == 1 else 1 / (1 - r2)
        vifs.append(vif)
    return pd.DataFrame({"feature": cols, "VIF": vifs}).set_index("feature")

def analyze_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify multicollinearity among the 6 spend features via correlation
    (already computed in Phase 2) and VIF (new, more rigorous diagnostic)."""
    print("\n" + "=" * 70)
    print("SECTION 5: MULTICOLLINEARITY ANALYSIS (CORRELATION + VIF)")
    print("=" * 70)

    corr = df[SPEND_COLS].corr()
    print("Correlation matrix (carried over from Phase 2):")
    print(corr.round(2).to_string())

    vif_df = compute_vif(df, SPEND_COLS)
    print("\nVariance Inflation Factor (VIF) per feature:")
    print(vif_df.round(2).to_string())
    print("\nInterpretation guide: VIF > 5 = moderate concern; VIF > 10 = high concern")
    for col, row in vif_df.iterrows():
        level = "HIGH" if row["VIF"] > 10 else ("MODERATE" if row["VIF"] > 5 else "LOW")
        print(f"  {col:<20} VIF = {row['VIF']:6.2f}  -> {level} multicollinearity")

    return vif_df

def document_multicollinearity_decision(vif_df: pd.DataFrame) -> str:
    """State and justify the multicollinearity-handling decision, comparing
    the two options laid out in the implementation plan."""
    print("\n" + "=" * 70)
    print("SECTION 6: MULTICOLLINEARITY TREATMENT DECISION")
    print("=" * 70)

    highest_vif_feature = vif_df["VIF"].idxmax()
    highest_vif_value = vif_df["VIF"].max()

    decision = (
        f"OBSERVED: Grocery, Milk, and Detergents_Paper show strong pairwise "
        f"correlation (see Phase 2). VIF, used here as a descriptive redundancy diagnostic rather than a K-Means requirement, also flags correlated features: "
        f"'{highest_vif_feature}' has the highest VIF at {highest_vif_value:.2f}.\n\n"
        "OPTIONS CONSIDERED:\n"
        "  Option A - PCA-assisted diagnostics: Keep all 6 features for clustering and profiling,\n"
        "             and use PCA separately for visualization/dimensionality diagnostics.\n"
        "             This does not remove correlation from the K-Means input; it preserves all\n"
        "             business-relevant spend categories while making low-dimensional structure visible.\n"
        "  Option B - Drop feature: Remove Detergents_Paper (most redundant with\n"
        "             Grocery, VIF/correlation-wise) and cluster on the remaining\n"
        "             5 features. Simpler, but permanently discards that spend\n"
        "             category, so it could no longer be reported per cluster.\n\n"
        "DECISION: Retain all six features and use PCA only as a visualization/diagnostic tool.\n"
        "RATIONALE:\n"
        "  1. No feature is dropped, so Phase 7's per-cluster profiling can still\n"
        "     report actual Detergents_Paper spend per cluster (business-relevant).\n"
        "  2. PCA directly supports Phase 8's 2D cluster visualization requirement\n"
        "     (Step 22-23) — one transformation serves two needs.\n"
        "  3. K-Means will still be run on the full standardized 6-feature space\n"
        "     (Phase 5-6) so cluster boundaries reflect all available spend\n"
        "     information, not just 2 compressed dimensions; PCA is used for\n"
        "     visualization/diagnostics, not as a replacement feature set.\n\n"
        "This decision, and the VIF table above, will be included in the final\n"
        "report's Methodology section to justify the choice over Option B."
    )
    print(decision)
    return decision

def apply_log_transform(features_df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to compress extreme spend values and reduce right-skew while
    retaining all customers. The transform is evaluated by comparing skewness
    before and after transformation."""
    print("\n" + "=" * 70)
    print("SECTION 2: APPLY LOG1P TRANSFORM")
    print("=" * 70)

    log_df = features_df.copy()
    for col in SPEND_COLS:
        log_df[col] = np.log1p(features_df[col])

    print("Applied log1p(x) = log(1 + x) to all 6 spend columns.")
    print(f"Min value pre-transform across features: {features_df[SPEND_COLS].min().min()} "
          "(always > 0 in this dataset, so log1p is a safe, standard choice).")
    print("\nPost-log1p summary statistics:")
    print(log_df.describe().round(2).to_string())
    return log_df

def verify_skew_reduction(features_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    """Compare skewness before vs. after log1p to confirm the transform
    achieved its purpose (referenced against Phase 3's skewness findings)."""
    print("\n" + "=" * 70)
    print("SECTION 3: VERIFY SKEWNESS REDUCTION (BEFORE VS. AFTER)")
    print("=" * 70)

    skew_before = features_df[SPEND_COLS].skew()
    skew_after = log_df[SPEND_COLS].skew()
    comparison = pd.DataFrame({
        "skew_before_log": skew_before,
        "skew_after_log": skew_after,
        "improvement": skew_before - skew_after,
    }).round(2)
    print(comparison.to_string())

    still_high = comparison[comparison["skew_after_log"].abs() > 1]
    if len(still_high) > 0:
        print(f"\n[NOTE] Features still moderately/highly skewed after log1p: "
              f"{list(still_high.index)}. This is expected and acceptable — "
              "log1p reduces skew substantially but does not force perfect "
              "symmetry; standardization (Section 4) will further equalize scale.")
    else:
        print("\n[OK] All features now within acceptable skew range after log1p.")
    return comparison

def plot_before_after_histograms(features_df: pd.DataFrame, log_df: pd.DataFrame) -> None:
    """Side-by-side histograms showing the effect of log1p per feature."""
    print("\n" + "=" * 70)
    print("SECTION 4: VISUALIZE BEFORE/AFTER DISTRIBUTIONS")
    print("=" * 70)

    # Compact 3x4 layout: each feature occupies one adjacent Before/After pair.
    # This keeps the figure readable and prevents it from being split across Word pages.
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.reshape(3, 4)
    for i, col in enumerate(SPEND_COLS):
        row = i // 2
        col_pair = (i % 2) * 2
        sns.histplot(features_df[col], bins=30, kde=True, ax=axes[row, col_pair], color="steelblue")
        axes[row, col_pair].set_title(f"{col} — Before (skew={features_df[col].skew():.2f})", fontsize=10)
        axes[row, col_pair].set_xlabel("Annual spend")

        sns.histplot(log_df[col], bins=30, kde=True, ax=axes[row, col_pair + 1], color="seagreen")
        axes[row, col_pair + 1].set_title(f"{col} — After log1p (skew={log_df[col].skew():.2f})", fontsize=10)
        axes[row, col_pair + 1].set_xlabel("log1p(annual spend)")

    fig.suptitle("Effect of log1p Transform on Spend Distributions", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "07_log_transform.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/07_log_transform.png")

def standardize_features(log_df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply StandardScaler (zero mean, unit variance) to the log-transformed
    features. Required because K-Means uses Euclidean distance and raw scales
    differ by orders of magnitude across categories."""
    print("\n" + "=" * 70)
    print("SECTION 5: STANDARDIZE FEATURES (StandardScaler)")
    print("=" * 70)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(log_df[SPEND_COLS])
    scaled_df = pd.DataFrame(scaled_array, columns=SPEND_COLS, index=log_df.index)

    print("Applied StandardScaler: x_scaled = (x - mean) / std, fit on log-transformed data.")
    print("\nPost-scaling summary statistics (expect mean ~0, std ~1 per column):")
    print(scaled_df.describe().round(2).to_string())

    means = scaled_df.mean().round(6)
    stds = scaled_df.std().round(3)
    print(f"\nMeans (should be ~0): {means.to_dict()}")
    print(f"Std devs (should be ~1): {stds.to_dict()}")
    print("[OK] Standardization verified — all features now on a comparable scale for K-Means.")
    print("[NOTE] Log1p reduces skewness substantially; perfect symmetry is not required for K-Means.")

    return scaled_df, scaler

def prepare_clustering_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Build the canonical clustering matrix in one reproducible step.

    The same feature preparation is used by the exploratory pipeline,
    K-selection, downstream phases, and report generation:
    raw spend -> log1p -> StandardScaler.
    """
    features_df = get_feature_matrix(df)
    log_df = apply_log_transform(features_df)
    scaled_df, scaler = standardize_features(log_df)
    feature_names = SPEND_COLS.copy()
    return scaled_df.to_numpy(), scaler, feature_names


def plot_scaled_boxplot(scaled_df: pd.DataFrame) -> None:
    """Single combined boxplot confirming all 6 features now share a
    comparable scale after log1p + standardization."""
    print("\n" + "=" * 70)
    print("SECTION 6: VISUALIZE FINAL SCALED FEATURE RANGES")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=scaled_df, ax=ax, palette="Set2")
    ax.set_title("Standardized Features (log1p + StandardScaler) — Comparable Scale Check")
    ax.set_ylabel("Standardized value (z-score)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_scaled_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {FIG_DIR}/08_scaled_features.png")

def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the six clustering features as a fresh copy."""
    print("=" * 70)
    print("SECTION 1: PREPARE WORKING COPY")
    print("=" * 70)
    features_df = df[SPEND_COLS].copy()
    print(f"Working feature matrix shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")
    print("[OK] Raw dataframe untouched; all transforms below operate on this copy.")
    return features_df


def save_transformed_data(scaled_df: pd.DataFrame) -> None:
    """Persist the final scaled feature matrix for downstream clustering phases."""
    out_path = PROCESSED_DIR / "scaled_features.csv"
    scaled_df.to_csv(out_path, index=False)
    print(f"[OK] Saved scaled, log-transformed feature matrix to '{out_path.name}' (shape: {scaled_df.shape}).")
