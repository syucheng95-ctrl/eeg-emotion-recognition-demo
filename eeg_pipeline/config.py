from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Band:
    name: str
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 250
    train_trial_samples: int = 12500
    test_trial_samples: int = 2500
    train_trials_per_class: int = 4
    test_trials_per_subject: int = 8
    window_seconds: float = 2.0
    step_seconds: float = 2.0
    patch_seconds: float = 0.5
    patch_step_seconds: float = 0.5
    aggregation: str = "mean"
    channel_zscore: bool = False
    baseline_seconds: float = 0.0
    denoise_low_hz: float = 1.0
    denoise_high_hz: float = 45.0
    bands: tuple[Band, ...] = field(
        default_factory=lambda: (
            Band("theta", 4.0, 8.0),
            Band("alpha", 8.0, 14.0),
            Band("beta", 14.0, 31.0),
            Band("gamma", 31.0, 50.0),
        )
    )

    @property
    def window_samples(self) -> int:
        return int(self.window_seconds * self.sample_rate)

    @property
    def step_samples(self) -> int:
        return int(self.step_seconds * self.sample_rate)

    @property
    def baseline_samples(self) -> int:
        return int(self.baseline_seconds * self.sample_rate)

    @property
    def patch_samples(self) -> int:
        return int(self.patch_seconds * self.sample_rate)

    @property
    def patch_step_samples(self) -> int:
        return int(self.patch_step_seconds * self.sample_rate)


def get_band_preset(name: str) -> tuple[Band, ...]:
    presets = {
        "base4": (
            Band("theta", 4.0, 8.0),
            Band("alpha", 8.0, 14.0),
            Band("beta", 14.0, 31.0),
            Band("gamma", 31.0, 50.0),
        ),
        "classic5": (
            Band("delta", 1.0, 4.0),
            Band("theta", 4.0, 8.0),
            Band("alpha", 8.0, 14.0),
            Band("beta", 14.0, 31.0),
            Band("gamma", 31.0, 50.0),
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown band preset: {name}")
    return presets[name]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_root(self) -> Path:
        return self.root / "官方数据集"

    @property
    def train_root(self) -> Path:
        return self.data_root / "训练集"

    @property
    def test_root(self) -> Path:
        return self.data_root / "公开测试集"

    @property
    def template_path(self) -> Path:
        return self.root / "测试结果模板.xlsx"

    @property
    def artifacts_root(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs_root(self) -> Path:
        return self.root / "outputs_v2"

    @property
    def legacy_outputs_root(self) -> Path:
        return self.root / "outputs_legacy"
