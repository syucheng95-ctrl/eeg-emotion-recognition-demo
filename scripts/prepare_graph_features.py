from __future__ import annotations

import _bootstrap

import argparse
from pathlib import Path

from eeg_pipeline.config import FeatureConfig, Paths, get_band_preset
from eeg_pipeline.dataset import load_test_trials, load_train_trials
from eeg_pipeline.graph_features import build_trial_window_tensor, save_graph_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract graph-ready DE window tensors.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--band-preset", type=str, default="classic5", choices=["base4", "classic5"])
    parser.add_argument("--channel-zscore", action="store_true")
    parser.add_argument("--baseline-seconds", type=float, default=0.0)
    parser.add_argument("--cache-prefix", type=str, default="gnn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(args.root.resolve())
    cfg = FeatureConfig(
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        channel_zscore=args.channel_zscore,
        baseline_seconds=args.baseline_seconds,
        bands=get_band_preset(args.band_preset),
    )
    prefix = f"{args.cache_prefix}_"

    train_trials = load_train_trials(paths.train_root, cfg)
    train_tensors, train_meta = build_trial_window_tensor(train_trials, cfg)
    save_graph_cache(paths.artifacts_root / f"{prefix}train_graph.npz", train_tensors, train_meta, cfg)

    test_trials = load_test_trials(paths.test_root, cfg)
    test_tensors, test_meta = build_trial_window_tensor(test_trials, cfg)
    save_graph_cache(paths.artifacts_root / f"{prefix}test_graph.npz", test_tensors, test_meta, cfg)

    print(f"Saved train graph tensors: {train_tensors.shape}")
    print(f"Saved test graph tensors: {test_tensors.shape}")


if __name__ == "__main__":
    main()
