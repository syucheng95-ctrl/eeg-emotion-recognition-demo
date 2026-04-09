from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import FeatureConfig
from .dataset import TrialSample
from .preprocessing import apply_trial_preprocessing, apply_window_baseline


def _build_filter_bank(cfg: FeatureConfig) -> dict[str, np.ndarray]:
    filters: dict[str, np.ndarray] = {}
    nyquist = cfg.sample_rate / 2.0
    for band in cfg.bands:
        filters[band.name] = butter(
            N=4,
            Wn=[band.low_hz / nyquist, band.high_hz / nyquist],
            btype="bandpass",
            output="sos",
        )
    return filters


def _extract_window_de(window: np.ndarray, sos_bank: dict[str, np.ndarray]) -> np.ndarray:
    # DE for Gaussian variables is monotonic with log-variance, so we use that form.
    channel_features: list[np.ndarray] = []
    for band_name in sos_bank:
        filtered = sosfiltfilt(sos_bank[band_name], window, axis=1)
        variance = np.var(filtered, axis=1, ddof=1)
        de = 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(variance, 1e-8))
        channel_features.append(de.astype(np.float32))
    return np.stack(channel_features, axis=1)


def _iter_windows(signal: np.ndarray, window_samples: int, step_samples: int):
    max_start = signal.shape[1] - window_samples
    for start in range(0, max_start + 1, step_samples):
        end = start + window_samples
        yield signal[:, start:end]


def extract_trial_feature(
    signal: np.ndarray,
    cfg: FeatureConfig,
    sos_bank: dict[str, np.ndarray],
) -> np.ndarray:
    signal = apply_trial_preprocessing(signal, cfg)
    windows = list(_iter_windows(signal, cfg.window_samples, cfg.step_samples))
    if not windows:
        raise ValueError(
            f"Signal length {signal.shape[1]} is shorter than window size {cfg.window_samples}"
        )
    per_window = np.stack(
        [
            _extract_window_de(apply_window_baseline(window, cfg), sos_bank).reshape(-1)
            for window in windows
        ],
        axis=0,
    ).astype(np.float32)
    mean_feature = np.mean(per_window, axis=0, dtype=np.float32)
    if cfg.aggregation == "mean":
        return mean_feature
    if cfg.aggregation == "meanstd":
        std_feature = np.std(per_window, axis=0, dtype=np.float32)
        return np.concatenate([mean_feature, std_feature], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported aggregation mode: {cfg.aggregation}")


def build_feature_matrix(
    trials: list[TrialSample],
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    sos_bank = _build_filter_bank(cfg)
    features: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for trial in trials:
        features.append(extract_trial_feature(trial.signal, cfg, sos_bank))
        metadata.append(
            {
                "subject_id": trial.subject_id,
                "dataset_split": trial.dataset_split,
                "group_name": trial.group_name,
                "trial_id": trial.trial_id,
                "label": -1 if trial.label is None else trial.label,
            }
        )
    return np.stack(features, axis=0), metadata


def save_feature_cache(
    cache_path: Path,
    features: np.ndarray,
    metadata: list[dict[str, object]],
    cfg: FeatureConfig,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features.astype(np.float32),
        metadata=np.array(metadata, dtype=object),
        config=json.dumps(asdict(cfg), ensure_ascii=False),
    )


def load_feature_cache(cache_path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    with np.load(cache_path, allow_pickle=True) as data:
        features = data["features"].astype(np.float32)
        metadata = list(data["metadata"].tolist())
    return features, metadata
