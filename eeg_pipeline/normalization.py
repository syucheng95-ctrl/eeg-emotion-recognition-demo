from __future__ import annotations

import numpy as np


def fit_window_normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 1), dtype=np.float64)
    std = x.std(axis=(0, 1), dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_window_normalizer(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((x - mean[None, None, :, :]) / std[None, None, :, :]).astype(np.float32)
