from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as a CSV, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
