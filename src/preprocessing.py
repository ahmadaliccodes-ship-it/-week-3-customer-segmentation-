from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SPEND_COLS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def transform_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Apply log1p to spend features, then standardize them."""
    features = df[SPEND_COLS].copy()
    log_features = np.log1p(features)
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(log_features), columns=SPEND_COLS, index=df.index)
    return log_features, scaled, scaler
