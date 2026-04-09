from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np
import scipy.io as sio

from .config import FeatureConfig


@dataclass(frozen=True)
class TrialSample:
    subject_id: str
    dataset_split: str
    group_name: str
    trial_id: int
    label: int | None
    signal: np.ndarray


def _load_hdf5_matrix(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        data = np.array(handle[key], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D EEG matrix in {path}, got shape {data.shape}")
    # Official files store data as (time, channels).
    return data.T


def _load_mat_matrix(path: Path, key: str) -> np.ndarray:
    data_dict = sio.loadmat(path)
    if key not in data_dict:
        visible_keys = [item for item in data_dict.keys() if not item.startswith("__")]
        if len(visible_keys) == 1:
            key = visible_keys[0]
        else:
            raise KeyError(f"{key} not found in {path}. Available keys: {visible_keys}")
    data = np.array(data_dict[key], dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D EEG matrix in {path}, got shape {data.shape}")
    if data.shape[0] == 30:
        return data
    if data.shape[1] == 30:
        return data.T
    raise ValueError(f"Unexpected EEG matrix shape in {path}: {data.shape}")


def _load_matrix(path: Path, key: str) -> np.ndarray:
    try:
        return _load_hdf5_matrix(path, key)
    except OSError:
        return _load_mat_matrix(path, key)


def _iter_train_files(train_root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for group_dir in sorted(train_root.iterdir()):
        if not group_dir.is_dir():
            continue
        for mat_path in sorted(group_dir.glob("*.mat")):
            pairs.append((mat_path, group_dir.name))
    return pairs


def load_train_trials(train_root: Path, cfg: FeatureConfig) -> list[TrialSample]:
    trials: list[TrialSample] = []
    for mat_path, group_name in _iter_train_files(train_root):
        subject_id = mat_path.stem.replace("timedata", "")
        neu = _load_matrix(mat_path, "EEG_data_neu")
        pos = _load_matrix(mat_path, "EEG_data_pos")
        for class_name, label, data in (("neu", 0, neu), ("pos", 1, pos)):
            expected = cfg.train_trial_samples * cfg.train_trials_per_class
            if data.shape[1] != expected:
                raise ValueError(
                    f"{mat_path} {class_name} expected {expected} samples, got {data.shape[1]}"
                )
            for trial_idx in range(cfg.train_trials_per_class):
                start = trial_idx * cfg.train_trial_samples
                end = start + cfg.train_trial_samples
                trials.append(
                    TrialSample(
                        subject_id=subject_id,
                        dataset_split="train",
                        group_name=group_name,
                        trial_id=label * cfg.train_trials_per_class + trial_idx + 1,
                        label=label,
                        signal=data[:, start:end],
                    )
                )
    return trials


def _test_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1)), path.stem
    return 0, path.stem


def load_test_trials(test_root: Path, cfg: FeatureConfig) -> list[TrialSample]:
    trials: list[TrialSample] = []
    for mat_path in sorted(test_root.glob("*.mat"), key=_test_sort_key):
        subject_id = mat_path.stem
        data = _load_matrix(mat_path, subject_id)
        expected = cfg.test_trial_samples * cfg.test_trials_per_subject
        if data.shape[1] != expected:
            raise ValueError(f"{mat_path} expected {expected} samples, got {data.shape[1]}")
        for trial_idx in range(cfg.test_trials_per_subject):
            start = trial_idx * cfg.test_trial_samples
            end = start + cfg.test_trial_samples
            trials.append(
                TrialSample(
                    subject_id=subject_id,
                    dataset_split="test",
                    group_name="unknown",
                    trial_id=trial_idx + 1,
                    label=None,
                    signal=data[:, start:end],
                )
            )
    return trials
