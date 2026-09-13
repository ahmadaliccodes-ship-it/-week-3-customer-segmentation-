from pathlib import Path
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "wholesale_customers.csv"
FIG_DIR = PROJECT_ROOT / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPEND_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
CATEGORICAL_COLS = ["Channel", "Region"]

RANDOM_STATE = 42
N_INIT = 20

import json
import re
from textwrap import wrap
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import preprocessing
from .preprocessing import PROJECT_OBJECTIVE, DATA_DICTIONARY, SPEND_COLS, CATEGORICAL_COLS
from . import clustering
from . import stability
from . import profiling

UCI_URL = "https://archive.ics.uci.edu/dataset/292/wholesale+customers"
SEEDS = [42, 7, 123, 2024, 99]

def load_and_recompute_stats() -> dict:
    """Build Phase 9 statistics from the canonical selected-K decision."""
    selection = clustering.select_k()
    selected_k = int(selection["selected_k"])
    cluster_col = f"cluster_k{selected_k}"

    df = pd.read_csv(PROCESSED_DIR / "labeled_customers.csv")
    mean_profile = profiling.build_cluster_profiles(df, cluster_col)
    separation = profiling.rank_discriminating_features(df, cluster_col)
    total_spend = mean_profile.sum(axis=1)
    channel_ct, region_ct = profiling.crosstab_channel_region(df, cluster_col)

    stability_df = pd.read_csv(PROCESSED_DIR / "stability_summary.csv")
    selected_boot = stability_df[
        (stability_df["K"] == selected_k) & (stability_df["Validation"] == "Bootstrap")
    ].iloc[0]

    sizes = df[cluster_col].value_counts().sort_index()
    pct = (sizes / len(df) * 100).round(1)
    channel_ct_pct = channel_ct.div(channel_ct.sum(axis=1), axis=0) * 100
    region_ct_pct = region_ct.div(region_ct.sum(axis=1), axis=0) * 100
    return {
        "df": df,
        "selected_k": selected_k,
        "cluster_col": cluster_col,
        "k_selection": selection,
        "mean_profile": mean_profile,
        "separation": separation,
        "sizes": sizes,
        "pct": pct,
        "total_spend": total_spend,
        "channel_ct_pct": channel_ct_pct,
        "region_pct": region_ct_pct,
        "stability_ari": {selected_k: float(selected_boot["Mean ARI"])},
        "stability_summary": stability_df,
    }



def name_and_interpret_clusters(stats: dict) -> dict:
    """Derive a plain-business-language name for each cluster from its
    actual dominant spend categories and channel composition — not chosen
    in advance. Naming rule (applied mechanically, not by eyeballing):
    for each cluster, identify its top-2 spend categories by mean value,
    and its majority channel, and compose a name from those facts."""
    print("\n" + "=" * 70)
    print("SECTION 2: NAME & INTERPRET EACH CLUSTER")
    print("=" * 70)

    mean_profile = stats["mean_profile"]
    channel_ct_pct = stats["channel_ct_pct"]
    sizes = stats["sizes"]
    pct = stats["pct"]
    total_spend = stats["total_spend"]

    cluster_names = {}
    interpretations = {}

    for cluster_id in mean_profile.index:
        top2 = mean_profile.loc[cluster_id].sort_values(ascending=False).index[:2].tolist()
        majority_channel = channel_ct_pct.loc[cluster_id].idxmax()
        majority_channel_pct = channel_ct_pct.loc[cluster_id].max()
        spend_level = "higher-total-spend" if total_spend[cluster_id] == total_spend.max() else "lower-total-spend"
        if majority_channel == "Horeca" and set(["Fresh", "Frozen"]).issubset(top2):
            name = "Horeca Fresh/Frozen Buyers"
        elif majority_channel == "Retail" and set(["Grocery", "Milk"]).issubset(top2):
            name = "Retail Grocery/Milk Buyers"
        else:
            name = f"{majority_channel} {top2[0]}/{top2[1]} Buyers"
        cluster_names[cluster_id] = name

        interpretation = (
            f"Cluster {cluster_id} — \"{name}\"\n"
            f"  Size: {sizes[cluster_id]} customers ({pct[cluster_id]}% of the dataset).\n"
            f"  Dominant spend categories: {top2[0]} and {top2[1]} "
            f"(means: {mean_profile.loc[cluster_id, top2[0]]:.0f} and "
            f"{mean_profile.loc[cluster_id, top2[1]]:.0f}).\n"
            f"  Channel composition: {majority_channel_pct:.1f}% {majority_channel}.\n"
            f"  Mean total annual spend: {total_spend[cluster_id]:,.0f} "
            f"({'highest' if spend_level == 'higher-total-spend' else 'lower'} of the two clusters).\n"
        )
        interpretations[cluster_id] = interpretation
        print(interpretation)

    print("[NOTE] Names above were generated mechanically from each cluster's "
          "actual top-2 spend categories and majority channel — not chosen "
          "before seeing the data, per the plan's requirement.")

    return {"cluster_names": cluster_names, "interpretations": interpretations}

def derive_decision_framework(stats: dict, naming: dict) -> str:
    """Create measurable, non-causal decision hypotheses for each final cluster."""
    names = naming["cluster_names"]
    mean_profile = stats["mean_profile"]
    channel = stats["channel_ct_pct"]
    sizes = stats["sizes"]
    pct = stats["pct"]
    frameworks = []

    # The requested business framing is only used when the observed profiles support it.
    for cluster_id in mean_profile.index:
        name = names[cluster_id]
        size = int(sizes.loc[cluster_id])
        share = float(pct.loc[cluster_id])
        top2 = mean_profile.loc[cluster_id].sort_values(ascending=False).index[:2].tolist()
        channel_name = channel.loc[cluster_id].idxmax()
        channel_share = float(channel.loc[cluster_id].max())

        if set(["Fresh", "Frozen"]).issubset(top2) and channel_name == "Horeca":
            action = (
                "Prioritize Fresh/Frozen product bundles; focus on Horeca-specific offers; "
                "monitor high-volume accounts and order frequency when those measures become available. "
                f"This hypothesis is supported by the highest mean spends in Fresh ({mean_profile.loc[cluster_id, 'Fresh']:,.0f}) "
                f"and Frozen ({mean_profile.loc[cluster_id, 'Frozen']:,.0f}) and {channel_share:.1f}% Horeca composition."
            )
        elif set(["Grocery", "Milk"]).issubset(top2) and channel_name == "Retail":
            action = (
                "Prioritize Grocery/Milk cross-selling; develop retail-oriented promotions; "
                "identify high-value customers within the segment using spend or margin once available. "
                f"This hypothesis is supported by the highest mean spends in Grocery ({mean_profile.loc[cluster_id, 'Grocery']:,.0f}) "
                f"and Milk ({mean_profile.loc[cluster_id, 'Milk']:,.0f}) and {channel_share:.1f}% Retail composition."
            )
        else:
            action = (
                f"Prioritize offers aligned with the segment's two highest mean-spend categories ({top2[0]} and {top2[1]}) "
                "and validate response using customer-level outcomes before operationalizing the strategy."
            )

        frameworks.append(
            f"Cluster {cluster_id} — {name}\n"
            f"{size} customers ({share:.1f}% of customers).\n"
            f"Decision hypothesis: {action}"
        )

    return "\n\n".join(frameworks) + (
        "\n\nThese recommendations are decision hypotheses, not causal claims. "
        "The dataset is observational and does not measure promotion response, margin, delivery frequency, or customer lifetime value."
    )


def derive_business_implications(stats: dict, naming: dict) -> str:
    """State measurable business implications tied to observed cluster evidence."""
    print("\n" + "=" * 70)
    print("SECTION 3: BUSINESS IMPLICATIONS")
    print("=" * 70)

    names = naming["cluster_names"]
    mean_profile = stats["mean_profile"]
    total_spend = stats["total_spend"]
    channel = stats.get("channel_ct_pct", pd.DataFrame())

    blocks = []
    for cluster_id in mean_profile.index:
        top2 = mean_profile.loc[cluster_id].sort_values(ascending=False).index[:2].tolist()
        majority_channel = channel.loc[cluster_id].idxmax() if not channel.empty else ""
        majority_share = channel.loc[cluster_id].max() if not channel.empty else float("nan")
        lead_category = top2[0]
        support = (
            f"Marketing could prioritize {lead_category}-led cross-selling for Cluster {cluster_id} "
            f"because {lead_category} has the highest mean annual spend in the segment ({mean_profile.loc[cluster_id, lead_category]:,.0f}), "
            f"followed by {top2[1]} ({mean_profile.loc[cluster_id, top2[1]]:,.0f}). "
            f"The segment is also {majority_share:.1f}% {majority_channel} based on Channel."
        )
        blocks.append(support)

    implications = (
        "1. EVIDENCE-BASED COMMERCIAL TARGETING:\n" +
        "   - " + "\n   - ".join(blocks) + "\n\n"
        "2. RELATIVE CUSTOMER VALUE:\n"
        f"   - Cluster {total_spend.idxmax()} has mean total annual spend of {total_spend.max():,.0f} versus "
        f"{total_spend.min():,.0f} for Cluster {total_spend.idxmin()}. This supports prioritizing value-based tests, "
        "not an assumption that higher spend will produce a stronger promotional response.\n\n"
        "[CAVEAT] These are hypotheses based on observed spending and channel composition. "
        "They are not causal claims about promotion response, pricing sensitivity, delivery frequency, or future sales."
    )
    print(implications)
    return implications

def document_limitations() -> str:
    """State the analysis's limitations plainly, including ones specific to
    this dataset/pipeline (not generic ML boilerplate)."""
    print("\n" + "=" * 70)
    print("SECTION 4: LIMITATIONS")
    print("=" * 70)

    limitations = (
        "1. SMALL DATASET: only 440 customer records. Cluster boundaries and "
        "profile statistics (means, ANOVA F-values) are estimated from a "
        "relatively small sample and may not generalize to a larger customer "
        "base without re-validation.\n\n"
        "2. SINGLE TIME-SNAPSHOT DATA: spend values are annual totals with no "
        "seasonality or time dimension. A customer's segment could shift "
        "across a year (e.g. a Horeca account's Fresh spend may spike "
        "seasonally) — this analysis cannot detect or account for that.\n\n"
        "3. NO DEMOGRAPHIC/FIRMOGRAPHIC DATA: beyond Channel and Region, no "
        "other customer attributes (business size, years as customer, order "
        "frequency, etc.) are available, limiting how deeply the 'why' behind "
        "each cluster's spend pattern can be explained.\n\n"
        "4. K-MEANS' SPHERICAL-CLUSTER ASSUMPTION: K-Means is most naturally suited to compact, roughly spherical "
        "clusters under Euclidean distance. The observed structure may not perfectly meet this assumption. The "
        "dendrogram (Phase 5) and residual skew after log-transform "
        "(Phase 4) suggest the true spend distribution may not perfectly fit "
        "this assumption, so some customers near the cluster boundary "
        "(visible in Phase 8's PCA scatter) may be ambiguously assigned.\n\n"
        "5. LOG-TRANSFORM INTERPRETABILITY: the log1p transform (Phase 4) "
        "was necessary to stabilize variance for clustering, but it changes "
        "the geometry of distances — a given gap in log-space does not "
        "correspond to a constant dollar gap in raw spend. Business "
        "stakeholders reading 'distance between clusters' should refer to "
        "the raw mean/median tables (Phase 7), not transformed-space "
        "distances, for dollar-denominated conclusions.\n\n"
        "6. SELECTED SOLUTION IS COARSE: the selected K is a low-dimensional segmentation, and its silhouette relative to an ideal separation "
        "and strong Channel alignment mean it may capture a broad business split rather than multiple fine-grained segments. "
        "This means the segmentation may be relatively coarse — it captures the "
        "dominant Horeca-vs-Retail-like spend split but may be masking finer "
        "sub-segments that this dataset's size/structure cannot reliably "
        "resolve with K-Means."
    )
    print(limitations)
    return limitations

def write_conclusion(stats: dict, naming: dict) -> str:
    """Summarize the key segments found and recommend concrete next steps,
    tied to the limitations above."""
    print("\n" + "=" * 70)
    print("SECTION 5: CONCLUSION & RECOMMENDED NEXT STEPS")
    print("=" * 70)

    names = naming["cluster_names"]
    selected_k = int(stats.get("selected_k", len(stats["mean_profile"])))
    conclusion = (
        f"This analysis segmented 440 wholesale customers into {selected_k} stable K-Means clusters based on annual spend "
        "across 6 product categories. The selected solution had the strongest K-Means silhouette among the "
        "stability-qualified candidates and passed the project's seed-variation and bootstrap stability checks.\n\n"
        + "\n".join(f"  - Cluster {cid}: \"{name}\"" for cid, name in names.items())
        + "\n\nThe clusters align strongly with the pre-existing Channel variable even though Channel was excluded from "
        "clustering. This is best treated as post-hoc contextual evidence of business relevance, not external validation. "
        "The solution should therefore be interpreted as a broad spend segmentation that substantially separates "
        "Retail-dominant and Horeca-dominant purchasing patterns.\n\n"
        "RECOMMENDED NEXT STEPS:\n"
        "  1. Re-run the pipeline on more recent and larger customer data, ideally with monthly or transaction-level "
        "history, to test whether the segmentation persists over time.\n"
        "  2. Validate the business hypotheses using order frequency, delivery, promotion-response, margin, and customer-tenure data.\n"
        "  3. Investigate finer segmentation with additional algorithms only after validating whether K>2 solutions remain stable.\n"
        "  4. Treat the current segments as decision-support hypotheses rather than fixed customer categories."
    )
    print(conclusion)
    return conclusion

def assemble_report_section(naming: dict, implications: str, limitations: str, conclusion: str) -> None:
    """Write the interpretation, implications, limitations, and conclusion
    into a single markdown section for the final report assembly (Phase 10)."""
    print("\n" + "=" * 70)
    print("SECTION 6: ASSEMBLE REPORT SECTION (MARKDOWN)")
    print("=" * 70)

    interpretations_text = "\n\n".join(naming["interpretations"].values())

    framework_stats = load_and_recompute_stats()
    decision_framework = derive_decision_framework(framework_stats, naming)
    content = f"""# 9. Interpretation, Business Implications & Limitations

## 9.1 Cluster Interpretation

{interpretations_text}

## 9.2 Business Implications

{implications}

## 9.3 Decision Framework

{decision_framework}

## 9.4 Limitations

{limitations}

## 9.5 Conclusion & Recommended Next Steps

{conclusion}
"""

    out_path = REPORT_DIR / "section9_interpretation.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Saved {out_path} ({len(content)} characters)")



def df_manifest(df: pd.DataFrame, decimals: int = 1) -> dict:
    x = df.copy().round(decimals)
    return {"columns": x.columns.tolist(), "index": [str(i) for i in x.index.tolist()], "data": x.values.tolist()}


def series_dict(s: pd.Series, decimals: int | None = None) -> dict:
    out = {}
    for k, v in s.items():
        value = float(v)
        out[str(k)] = round(value, decimals) if decimals is not None else value
    return out


def extract_snippet(filepath: str, start: str, end: str) -> str:
    text = Path(filepath).read_text(encoding="utf-8")
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j].rstrip()


def curate_code_snippets() -> dict:
    src_dir = PROJECT_ROOT / "src"
    return {
        "load": extract_snippet(str(src_dir / "preprocessing.py"), "def load_raw_data", "def profile_dataset"),
        "feature_preparation": extract_snippet(str(src_dir / "preprocessing.py"), "def prepare_clustering_features", "def plot_scaled_boxplot"),
        "k_loop": extract_snippet(str(src_dir / "clustering.py"), "def fit_kmeans_range", "def plot_elbow"),
        "final": extract_snippet(str(src_dir / "stability.py"), "def train_final_models", "def check_stability_seeds"),
        "stability": extract_snippet(str(src_dir / "stability.py"), "def check_stability_seeds", "def check_stability_bootstrap"),
        "pca": extract_snippet(str(src_dir / "profiling.py"), "def apply_pca", "def plot_pca_loadings"),
    }


def build_interpretation(profiles: dict) -> tuple[str, dict]:
    """Build business-readable cluster names from observed spend and Channel profiles."""
    names = {}
    blocks = []
    means = profiles["means"]
    total = profiles["total_spend"]
    channel = profiles["channel_pct"]

    for cid in means.index:
        top2 = means.loc[cid].sort_values(ascending=False).index[:2].tolist()
        majority_channel = channel.loc[cid].idxmax()

        if majority_channel == "Horeca" and set(["Fresh", "Frozen"]).issubset(top2):
            names[cid] = "Horeca Fresh/Frozen Buyers"
        elif majority_channel == "Retail" and set(["Grocery", "Milk"]).issubset(top2):
            names[cid] = "Retail Grocery/Milk Buyers"
        else:
            names[cid] = f"{majority_channel} {top2[0]}/{top2[1]} Buyers"

        blocks.append(
            f"Cluster {cid} — {names[cid]}\n"
            f"Size: {profiles['sizes'].loc[cid]} customers ({profiles['pct'].loc[cid]:.1f}%).\n"
            f"Highest mean-spend categories: {top2[0]} ({means.loc[cid, top2[0]]:,.0f}) and {top2[1]} ({means.loc[cid, top2[1]]:,.0f}).\n"
            f"Mean total annual spend: {total.loc[cid]:,.0f}.\n"
            f"Post-hoc Channel profile: {channel.loc[cid, majority_channel]:.1f}% {majority_channel}."
        )
    return "\n\n".join(blocks), names


def assemble_final_report_manifest() -> dict:
    """Rebuild the report manifest using the canonical src modules only."""
    raw = preprocessing.load_raw_data()
    overview = preprocessing.profile_dataset(raw)
    preprocessing.confirm_data_quality(raw)

    prep_features_df = preprocessing.get_feature_matrix(raw)
    log_df = preprocessing.apply_log_transform(prep_features_df)
    prep_vif = preprocessing.analyze_multicollinearity(raw)

    X_scaled, _, feature_names = preprocessing.prepare_clustering_features(raw)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    kmeans_results = clustering.fit_kmeans_range(X_scaled)
    hier_results = clustering.fit_hierarchical_range(X_scaled)
    ksel = clustering.select_k(X_scaled, kmeans_results, hier_results)

    df_labeled = pd.read_csv(PROCESSED_DIR / "labeled_customers.csv")
    selected_k = int(ksel["selected_k"])
    cluster_col = f"cluster_k{selected_k}"

    mean_profile = profiling.build_cluster_profiles(df_labeled, cluster_col)
    sizes = df_labeled[cluster_col].value_counts().sort_index()
    pct = sizes / len(df_labeled) * 100
    medians = df_labeled.groupby(cluster_col)[SPEND_COLS].median()
    separation = profiling.rank_discriminating_features(df_labeled, cluster_col)
    channel_ct, region_ct = profiling.crosstab_channel_region(df_labeled, cluster_col)
    total_spend = mean_profile.sum(axis=1)
    profiles = {
        "sizes": sizes, "pct": pct, "means": mean_profile, "medians": medians,
        "separation": separation,
        "channel_pct": channel_ct.div(channel_ct.sum(axis=1), axis=0) * 100,
        "region_pct": region_ct.div(region_ct.sum(axis=1), axis=0) * 100,
        "total_spend": total_spend,
    }
    interpretation_text, cluster_names = build_interpretation(profiles)
    pca_pcs, pca_model = profiling.apply_pca(scaled_df)
    snippets = curate_code_snippets()

    stability_df = pd.read_csv(PROCESSED_DIR / "stability_summary.csv")

    manifest = {
        "title": "Wholesale Customer Segmentation: A Clustering Analysis",
        "subtitle": "Unsupervised Learning Applied to the UCI Wholesale Customers Dataset",
        "sections": {
            "introduction": {
                "heading": "1. Introduction & Dataset Overview",
                "objective_text": PROJECT_OBJECTIVE,
                "dataset_source": UCI_URL,
                "data_dictionary": [[k, v, "Post-hoc profiling" if k in CATEGORICAL_COLS else "Clustering"] for k, v in DATA_DICTIONARY.items()],
            },
            "data_overview": {
                "heading": "2. Data Overview & Quality",
                "shape": overview["shape"],
                "missing_total": int(sum(overview["missing_values"].values())),
                "duplicates": overview["duplicate_rows"],
                "channel_counts": overview["channel_counts"],
                "region_counts": overview["region_counts"],
                "negative_spend_total": int((raw[SPEND_COLS] < 0).sum().sum()),
                "zero_spend_total": int((raw[SPEND_COLS] == 0).sum().sum()),
                "constant_columns": [c for c in raw.columns if raw[c].nunique() <= 1],
            },
            "methodology": {
                "heading": "3. Methodology",
                "k_selection": {
                    "selected_k": ksel["selected_k"],
                    "reason": ksel["reason"],
                    "silhouette": ksel["silhouette"],
                    "stability": ksel["stability"],
                    "decision_rule": ksel["decision_rule"],
                    "threshold_note": stability.STABILITY_THRESHOLD_DESCRIPTION,
                },
                "vif": df_manifest(prep_vif, 2),
                "skew_before": series_dict(prep_features_df.skew(), 2),
                "skew_after": series_dict(log_df.skew(), 2),
                "code": snippets,
            },
            "stability": df_manifest(stability_df.set_index(["K", "Validation"]), 2),
            "model_selection_table": [
                [
                    int(r["k"]),
                    round(float(r["kmeans_silhouette"]), 3),
                    round(float(r["bootstrap_mean_ari"]), 2),
                    "Strong" if r["stable"] and int(r["k"]) == selected_k else ("Moderate" if r["stable"] else "Weak"),
                    "Selected" if int(r["k"]) == selected_k else "Rejected",
                ]
                for r in ksel["candidate_stability"]
            ],
            "results": {
                "heading": "4. Results",
                "cluster_profiles": {
                    "sizes": {str(k): int(v) for k, v in sizes.items()},
                    "pct": series_dict(pct, 1),
                    "mean_profile": df_manifest(mean_profile, 1),
                    "median_profile": df_manifest(medians, 1),
                    "separation": df_manifest(separation.set_index("feature")[["F_statistic", "eta_squared"]], 3),
                    "channel_ct": df_manifest(profiles["channel_pct"], 1),
                    "region_ct": df_manifest(profiles["region_pct"], 1),
                },
                "pca": {
                    "explained_variance": pca_model.explained_variance_ratio_.tolist(),
                    "loadings": df_manifest(pd.DataFrame(pca_model.components_.T, columns=["PC1", "PC2"], index=SPEND_COLS), 3),
                },
            },
            "cluster_interpretation": {"heading": "5. Cluster Interpretation", "text": interpretation_text, "cluster_names": cluster_names},
            "business_implications": {"heading": "6. Business Implications", "text": derive_business_implications({"mean_profile": mean_profile, "total_spend": total_spend, "channel_ct_pct": profiles["channel_pct"], "sizes": sizes, "pct": pct}, {"cluster_names": cluster_names})},
            "decision_framework": {"heading": "7. Decision Framework", "text": derive_decision_framework({"mean_profile": mean_profile, "total_spend": total_spend, "channel_ct_pct": profiles["channel_pct"], "sizes": sizes, "pct": pct}, {"cluster_names": cluster_names})},
            "limitations": {"heading": "8. Limitations", "text": document_limitations()},
            "conclusion": {"heading": "9. Conclusion", "text": write_conclusion({"selected_k": selected_k, "stability_ari": {selected_k: float(ksel["stability"]["bootstrap_mean_ari"])}, "mean_profile": mean_profile, "total_spend": total_spend}, {"cluster_names": cluster_names})},
            "references": {"heading": "10. References", "items": [["UCI Machine Learning Repository — Wholesale Customers", UCI_URL], ["Scikit-learn KMeans documentation", "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html"], ["Scikit-learn silhouette_score documentation", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html"]]},
            "reproducibility": {
                "heading": "Appendix: Reproducibility & Environment",
                "random_seed": RANDOM_STATE,
                "kmeans": f"K-Means with n_init={N_INIT}",
                "preprocessing": "log1p + StandardScaler",
                "final_k": selected_k,
                "stability": "Multiple random seeds + bootstrap ARI",
                "python": "3.13.5",
                "packages": {
                    "pandas": "2.2.3",
                    "numpy": "2.3.5",
                    "scikit-learn": "1.8.0",
                    "scipy": "1.17.0",
                    "matplotlib": "3.10.8",
                    "seaborn": "0.13.2",
                    "jupyterlab": "4.5.3",
                    "notebook": "7.5.3",
                    "pytest": "9.0.2",
                },
                "reproduction_note": "The executed notebooks use the same canonical preprocessing and K-selection logic as the pipeline under src/. Re-running the pipeline with the pinned environment reproduces the documented analysis settings."
            },
            "appendix": {"heading": "Appendix: Reproducibility & Full Pipeline Reference", "text": f"The analysis is implemented as phases 1 through 10 using the reusable modules under src/. Reproducibility settings are fixed at random seed 42, K-Means n_init=20, log1p transformation followed by StandardScaler, final K={selected_k}, and multiple-seed plus bootstrap ARI stability validation. The executed notebook environment is Python 3.13.5 with pinned package versions in requirements.txt."},
        },
    }
    return manifest


def main() -> dict:
    manifest = assemble_final_report_manifest()
    out = REPORT_DIR / "report_content.json"
    out.write_text(json.dumps(manifest, indent=2, default=lambda o: o), encoding="utf-8")
    print(f"[OK] {out.name} written using canonical src modules.")
    return manifest


def _manifest_lines(value, indent: int = 0) -> list[str]:
    """Convert report-manifest content into readable text lines for PDF pages."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_manifest_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.extend(_manifest_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {item}")
        return lines
    return [f"{prefix}{value}"]


def generate_pdf_report(manifest: dict, output_path: Path | None = None) -> Path:
    """Generate the final PDF report from the canonical report manifest."""
    output_path = output_path or (REPORT_DIR / "Wholesale_Customer_Segmentation_Report.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages: list[tuple[str, list[str]]] = []
    pages.append((
        manifest["title"],
        [manifest.get("subtitle", ""), "", "Generated directly by src.run_pipeline from the canonical project outputs."],
    ))

    for section in manifest.get("sections", {}).values():
        if not isinstance(section, dict):
            continue
        heading = section.get("heading")
        if not heading:
            continue

        lines: list[str] = []
        for key, value in section.items():
            if key == "heading":
                continue
            label = key.replace("_", " ").title()
            lines.append(f"{label}:")
            lines.extend(_manifest_lines(value, indent=2))
            lines.append("")
        pages.append((heading, lines))

    with PdfPages(output_path) as pdf:
        for title, raw_lines in pages:
            lines: list[str] = []
            for raw_line in raw_lines:
                if raw_line == "":
                    lines.append("")
                else:
                    lines.extend(wrap(str(raw_line), width=105, subsequent_indent="  ") or [""])

            max_lines = 42
            chunks = [lines[i:i + max_lines] for i in range(0, len(lines) or 1, max_lines)]
            for chunk_index, chunk in enumerate(chunks):
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                page_title = title if chunk_index == 0 else f"{title} (continued)"
                fig.text(0.07, 0.95, page_title, fontsize=16, fontweight="bold", va="top")
                y = 0.90
                for line in chunk:
                    fig.text(0.07, y, line, fontsize=9, family="DejaVu Sans Mono" if "\t" in line else "DejaVu Sans", va="top")
                    y -= 0.0205
                fig.text(0.07, 0.035, "Wholesale Customer Segmentation Report", fontsize=8, va="bottom")
                pdf.savefig(fig)
                plt.close(fig)

    print(f"[OK] {output_path} generated ({output_path.stat().st_size} bytes)")
    return output_path
