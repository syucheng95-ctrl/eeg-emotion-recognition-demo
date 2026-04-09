from __future__ import annotations

import numpy as np

from .config import FeatureConfig


def apply_trial_preprocessing(signal: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    if not cfg.channel_zscore:
        return signal.astype(np.float32, copy=False)
    mean = signal.mean(axis=1, keepdims=True, dtype=np.float64)
    std = signal.std(axis=1, keepdims=True, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return ((signal - mean) / std).astype(np.float32)


def apply_window_baseline(window: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    if cfg.baseline_samples <= 0:
        return window
    baseline_samples = min(cfg.baseline_samples, window.shape[1])
    if baseline_samples <= 0:
        return window
    baseline = window[:, :baseline_samples].mean(axis=1, keepdims=True, dtype=np.float64)
    return (window - baseline).astype(np.float32)
