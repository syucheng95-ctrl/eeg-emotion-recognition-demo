from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import FeatureConfig
from .features import _build_filter_bank, _extract_window_de, _iter_windows
from .dataset import TrialSample
from .preprocessing import apply_trial_preprocessing, apply_window_baseline


def build_trial_window_tensor(
    trials: list[TrialSample],
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    sos_bank = _build_filter_bank(cfg)
    tensors: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for trial in trials:
        signal = apply_trial_preprocessing(trial.signal, cfg)
        windows = list(_iter_windows(signal, cfg.window_samples, cfg.step_samples))
        if not windows:
            raise ValueError(
                f"Signal length {signal.shape[1]} is shorter than window size {cfg.window_samples}"
            )
        window_features = np.stack(
            [_extract_window_de(apply_window_baseline(window, cfg), sos_bank) for window in windows],
            axis=0,
        ).astype(np.float32)
        tensors.append(window_features)
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


def save_graph_cache(
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


def load_graph_cache(cache_path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    with np.load(cache_path, allow_pickle=True) as data:
        tensors = data["tensors"].astype(np.float32)
        metadata = list(data["metadata"].tolist())
    return tensors, metadata
