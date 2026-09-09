from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import SPEND_COLS, transform_and_scale
from src.clustering import fit_kmeans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "wholesale_customers.csv"


def test_raw_dataset_shape_and_columns():
    df = pd.read_csv(DATA)
    assert df.shape == (440, 8)
    assert set(SPEND_COLS).issubset(df.columns)


def test_preprocessing_returns_standardized_features():
    df = pd.read_csv(DATA)
    _, scaled, _ = transform_and_scale(df)
    assert scaled.shape == (440, 6)
    assert scaled.mean().abs().max() < 1e-10
    assert (scaled.std(ddof=0) - 1).abs().max() < 1e-10


def test_kmeans_returns_two_clusters():
    df = pd.read_csv(DATA)
    _, scaled, _ = transform_and_scale(df)
    _, labels = fit_kmeans(scaled, k=2)
    assert len(set(labels)) == 2
