from __future__ import annotations

import numpy as np
import pandas as pd


def competition_rmse(y_true: pd.DataFrame | np.ndarray, y_pred: pd.DataFrame | np.ndarray) -> float:
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    weights = np.where(true_values >= 0.5, 1.2, 1.0)
    return float(np.sqrt(np.mean(weights * np.square(pred_values - true_values))))
