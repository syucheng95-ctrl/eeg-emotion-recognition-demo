from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import FeatureConfig
from .dataset import TrialSample
from .preprocessing import apply_trial_preprocessing


def _build_denoise_filter(cfg: FeatureConfig) -> np.ndarray:
    nyquist = cfg.sample_rate / 2.0
    low = max(cfg.denoise_low_hz / nyquist, 1e-5)
    high = min(cfg.denoise_high_hz / nyquist, 0.999)
    return butter(N=4, Wn=[low, high], btype="bandpass", output="sos")


def _iter_patches(signal: np.ndarray, patch_samples: int, step_samples: int):
    max_start = signal.shape[1] - patch_samples
    for start in range(0, max_start + 1, step_samples):
        end = start + patch_samples
        yield signal[:, start:end]


def build_raw_patch_tensor(
    trials: list[TrialSample],
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    sos = _build_denoise_filter(cfg)
    tensors: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for trial in trials:
        signal = apply_trial_preprocessing(trial.signal, cfg)
        signal = sosfiltfilt(sos, signal, axis=1).astype(np.float32)
        patches = list(_iter_patches(signal, cfg.patch_samples, cfg.patch_step_samples))
        if not patches:
            raise ValueError(
                f"Signal length {signal.shape[1]} is shorter than patch size {cfg.patch_samples}"
            )
        tensors.append(np.stack(patches, axis=0).astype(np.float32))
        metadata.append(
            {
                "subject_id": trial.subject_id,
                "dataset_split": trial.dataset_split,
                "group_name": trial.group_name,
                "trial_id": trial.trial_id,
                "label": -1 if trial.label is None else trial.label,
            }
        )
    return np.stack(tensors, axis=0), metadata


def save_raw_patch_cache(
    cache_path: Path,
    tensors: np.ndarray,
    metadata: list[dict[str, object]],
    cfg: FeatureConfig,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        tensors=tensors.astype(np.float32),
        metadata=np.array(metadata, dtype=object),
        config=json.dumps(asdict(cfg), ensure_ascii=False),
    )


def load_raw_patch_cache(cache_path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    with np.load(cache_path, allow_pickle=True) as data:
        tensors = data["tensors"].astype(np.float32)
        metadata = list(data["metadata"].tolist())
    return tensors, metadata
