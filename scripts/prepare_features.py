from __future__ import annotations

import _bootstrap

import argparse
from pathlib import Path

from eeg_pipeline.config import FeatureConfig, Paths, get_band_preset
from eeg_pipeline.dataset import load_test_trials, load_train_trials
from eeg_pipeline.features import build_feature_matrix, save_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DE trial-level features.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root path.")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "meanstd"])
    parser.add_argument("--band-preset", type=str, default="base4", choices=["base4", "classic5"])
    parser.add_argument("--channel-zscore", action="store_true")
    parser.add_argument("--baseline-seconds", type=float, default=0.0)
    parser.add_argument("--cache-prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(args.root.resolve())
    cfg = FeatureConfig(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        aggregation=args.aggregation,
        channel_zscore=args.channel_zscore,
        baseline_seconds=args.baseline_seconds,
        bands=get_band_preset(args.band_preset),
    )
    prefix = f"{args.cache_prefix}_" if args.cache_prefix else ""

    train_trials = load_train_trials(paths.train_root, cfg)
    train_features, train_meta = build_feature_matrix(train_trials, cfg)
    save_feature_cache(
        paths.artifacts_root / f"{prefix}train_features.npz",
        train_features,
        train_meta,
        cfg,
    )

    test_trials = load_test_trials(paths.test_root, cfg)
    test_features, test_meta = build_feature_matrix(test_trials, cfg)
    save_feature_cache(
        paths.artifacts_root / f"{prefix}test_features.npz",
        test_features,
        test_meta,
        cfg,
    )

    print(f"Saved train features: {train_features.shape}")
    print(f"Saved test features: {test_features.shape}")


if __name__ == "__main__":
    main()
