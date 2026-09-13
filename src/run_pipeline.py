"""Run the complete Week 3 customer-segmentation analysis from raw data.

Usage:
    python -m src.run_pipeline
"""

from pathlib import Path
import logging

from . import preprocessing
from . import clustering
from . import stability
from . import profiling
from . import reporting


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting Week 3 customer-segmentation pipeline")

    # Phase 1-2: load, validate, and explore the raw dataset.
    df_raw = preprocessing.load_raw_data(snapshot=False)
    preprocessing.profile_dataset(df_raw)
    preprocessing.confirm_data_quality(df_raw)
    assert not df_raw.isna().any().any(), "Raw dataset contains missing values."
    assert (df_raw[preprocessing.SPEND_COLS] >= 0).all().all(), "Spend features contain negative values."
    logger.info("Loaded dataset: %s rows, %s columns", *df_raw.shape)
    preprocessing.plot_histograms(df_raw)
    preprocessing.plot_boxplots(df_raw)
    preprocessing.plot_categorical_bars(df_raw)
    preprocessing.plot_correlation_heatmap(df_raw)
    preprocessing.plot_pairplot_by_channel(df_raw)
    features_df = preprocessing.select_clustering_features(df_raw)

    # Phase 3: skewness, outliers, and multicollinearity.
    preprocessing.analyze_skewness(df_raw)
    outlier_summary = preprocessing.detect_outliers_iqr(df_raw)
    preprocessing.document_outlier_decision()
    preprocessing.plot_outlier_reference(df_raw, outlier_summary)
    vif_df = preprocessing.analyze_multicollinearity(df_raw)
    preprocessing.document_multicollinearity_decision(vif_df)

    # Phase 4: canonical feature preparation (log1p + StandardScaler).
    X, _, feature_names = preprocessing.prepare_clustering_features(df_raw)
    assert X.shape[1] == 6, f"Expected 6 clustering features, got {X.shape[1]}"
    assert preprocessing.SPEND_COLS == feature_names, "Canonical clustering feature order changed."
    log_df = preprocessing.apply_log_transform(features_df)
    preprocessing.verify_skew_reduction(features_df, log_df)
    preprocessing.plot_before_after_histograms(features_df, log_df)
    scaled_df = __import__("pandas").DataFrame(X, columns=feature_names)
    preprocessing.plot_scaled_boxplot(scaled_df)
    preprocessing.save_transformed_data(scaled_df)

    # Phase 5: model comparison and canonical K selection.
    scaled_loaded = clustering.load_scaled_features()
    X = scaled_loaded[preprocessing.SPEND_COLS].values
    kmeans_results = clustering.fit_kmeans_range(X)
    clustering.plot_elbow(kmeans_results)
    hier_results = clustering.fit_hierarchical_range(X)
    clustering.plot_dendrogram(X)
    clustering.plot_silhouette_comparison(kmeans_results, hier_results)
    clustering.run_detailed_silhouette_for_top_candidates(X, kmeans_results)
    k_selection = clustering.select_k(X, kmeans_results, hier_results, force_recompute=True)
    final_k = int(k_selection["selected_k"])
    assert final_k >= 2, f"Selected K must be >= 2, got {final_k}"
    assert final_k <= 10, f"Selected K must be <= 10, got {final_k}"
    logger.info("Selected K=%s", final_k)

    # Phase 6: final models and stability validation.
    df_raw_stability, X_stability = stability.load_data()
    models = stability.train_final_models(X_stability)
    labeled_df = stability.assign_labels_to_raw(df_raw_stability, models)

    canonical_selection = clustering.select_k(X_stability)
    stability_results = stability.compute_stability_results(X_stability)
    selected_bootstrap_mean = stability.summarize_ari(
        stability_results[final_k]["boot_ari"]["ari_vs_baseline"]
    )["mean_ari"]
    logger.info("Bootstrap mean ARI=%.3f", selected_bootstrap_mean)

    stability.plot_stability_summary(stability_results)
    stability.document_stability_conclusion(stability_results)
    stability.save_labeled_data(labeled_df)
    stability.save_stability_summary(stability_results)

    # Phase 7: cluster profiling and business analysis.
    selected_k = int(canonical_selection["selected_k"])
    assert selected_k == final_k, "Canonical K changed between pipeline stages."
    primary_cluster_col = profiling.cluster_column(selected_k)
    labeled_loaded = profiling.load_labeled_data(selected_k)
    profiling.report_cluster_sizes(labeled_loaded, primary_cluster_col)
    mean_profile = profiling.build_cluster_profiles(
        labeled_loaded, primary_cluster_col
    )
    profiling.plot_cluster_profile_bars(mean_profile, primary_cluster_col)
    ranking = profiling.rank_discriminating_features(
        labeled_loaded, primary_cluster_col
    )
    profiling.plot_feature_ranking(ranking, primary_cluster_col)
    profiling.compare_high_low_spend_clusters(
        mean_profile, primary_cluster_col
    )
    channel_ct, region_ct = profiling.crosstab_channel_region(
        labeled_loaded, primary_cluster_col
    )
    profiling.plot_channel_crosstab(channel_ct, primary_cluster_col)
    profiling.plot_region_crosstab(region_ct, primary_cluster_col)
    profiling.brief_secondary_k3_summary(labeled_loaded, selected_k)

    # Phase 8: PCA visualization.
    scaled_pca, labeled_pca = profiling.load_pca_data()
    pcs, pca_model = profiling.apply_pca(scaled_pca)
    profiling.plot_pca_loadings(pca_model)
    profiling.plot_pca_cluster_scatter(pcs, labeled_pca, pca_model, primary_cluster_col)
    profiling.plot_pca_channel_overlay(pcs, labeled_pca, pca_model, primary_cluster_col)

    # Phase 9-10: business interpretation and report assembly.
    stats = reporting.load_and_recompute_stats()
    naming = reporting.name_and_interpret_clusters(stats)
    implications = reporting.derive_business_implications(stats, naming)
    limitations = reporting.document_limitations()
    conclusion = reporting.write_conclusion(stats, naming)
    reporting.assemble_report_section(
        naming, implications, limitations, conclusion
    )
    report_manifest = reporting.main()
    reporting.generate_pdf_report(report_manifest)

    logger.info("Pipeline complete")
    logger.info("Selected K=%s", k_selection["selected_k"])
    logger.info("Processed data: data/processed/")
    logger.info("Figures: figures/")
    logger.info("Report outputs: reports/")
    logger.info("Final report PDF: reports/Wholesale_Customer_Segmentation_Report.pdf")


if __name__ == "__main__":
    main()
