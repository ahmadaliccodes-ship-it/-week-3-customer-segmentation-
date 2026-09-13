from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from src import clustering, preprocessing, stability


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wholesale_customers.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "Wholesale_Customer_Segmentation_Report.pdf"

EXPECTED_COLUMNS = [
    "Channel",
    "Region",
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]


def test_dataset_has_expected_columns():
    df = pd.read_csv(RAW_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values():
    df = pd.read_csv(RAW_PATH)
    assert not df.isna().any().any()


def test_spend_features_are_non_negative():
    df = pd.read_csv(RAW_PATH)
    assert (df[preprocessing.SPEND_COLS] >= 0).all().all()


def test_dataset_has_expected_row_count():
    df = pd.read_csv(RAW_PATH)
    assert len(df) == 440


def test_scaled_features_have_expected_columns():
    scaled = pd.read_csv(PROCESSED_DIR / "scaled_features.csv")
    assert list(scaled.columns) == preprocessing.SPEND_COLS


def test_scaled_features_are_standardized():
    scaled = pd.read_csv(PROCESSED_DIR / "scaled_features.csv")
    np.testing.assert_allclose(scaled.mean().to_numpy(), 0.0, atol=1e-10)
    np.testing.assert_allclose(scaled.std(ddof=0).to_numpy(), 1.0, atol=1e-10)


def test_scaled_features_are_finite():
    scaled = pd.read_csv(PROCESSED_DIR / "scaled_features.csv")
    assert np.isfinite(scaled.to_numpy()).all()


def test_k_selection_is_valid_and_reproducible():
    selection = clustering.select_k()
    assert selection["selected_k"] in stability.CANDIDATE_KS
    assert selection["selected_k"] == 2
    assert selection["stability"]["pass"] is True


def test_final_cluster_count_matches_selected_k():
    selection = clustering.select_k()
    selected_k = int(selection["selected_k"])
    labeled = pd.read_csv(PROCESSED_DIR / "labeled_customers.csv")
    cluster_col = f"cluster_k{selected_k}"
    assert cluster_col in labeled.columns
    assert labeled[cluster_col].nunique() == selected_k
    assert set(labeled[cluster_col].dropna().unique()) == set(range(selected_k))


def test_stability_summary_contains_required_statistics():
    summary = pd.read_csv(PROCESSED_DIR / "stability_summary.csv")
    required_columns = {
        "K",
        "Validation",
        "Mean ARI",
        "Median ARI",
        "Std ARI",
        "Minimum ARI",
        "Maximum ARI",
        "25th Percentile",
        "75th Percentile",
        "% Runs Above Threshold",
        "Threshold",
        "Runs",
    }
    assert required_columns.issubset(summary.columns)
    assert not summary[list(required_columns)].isna().any().any()
    assert (summary["Threshold"] == 0.75).all()


def test_model_reproducibility():
    scaled = pd.read_csv(PROCESSED_DIR / "scaled_features.csv")
    X = scaled[preprocessing.SPEND_COLS].to_numpy()

    result1 = stability.train_final_models(X)
    result2 = stability.train_final_models(X)

    selected_k = int(clustering.select_k()["selected_k"])
    assert np.array_equal(
        result1[selected_k]["labels"],
        result2[selected_k]["labels"],
    )


def test_final_business_outputs_exist():
    expected_outputs = [
        PROCESSED_DIR / "scaled_features.csv",
        PROCESSED_DIR / "labeled_customers.csv",
        PROCESSED_DIR / "stability_summary.csv",
        REPORT_PATH,
    ]
    for output_path in expected_outputs:
        assert output_path.is_file(), f"Missing expected pipeline output: {output_path}"
        assert output_path.stat().st_size > 0, f"Pipeline output is empty: {output_path}"


def test_end_to_end_pipeline_generates_report_pdf(tmp_path):
    """Run the complete pipeline from a clean project copy and verify its PDF output."""
    project_copy = tmp_path / "week-3-customer-segmentation"
    shutil.copytree(
        PROJECT_ROOT,
        project_copy,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    # Remove generated artifacts so this test proves the pipeline can rebuild them from the raw dataset.
    for generated_dir in [
        project_copy / "data" / "processed",
        project_copy / "figures",
        project_copy / "reports",
    ]:
        for child in generated_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        generated_dir.mkdir(parents=True, exist_ok=True)

    report_path = project_copy / "reports" / "Wholesale_Customer_Segmentation_Report.pdf"

    result = subprocess.run(
        [sys.executable, "-m", "src.run_pipeline"],
        cwd=project_copy,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert report_path.is_file()
    assert report_path.stat().st_size > 0
    assert report_path.read_bytes().startswith(b"%PDF")
