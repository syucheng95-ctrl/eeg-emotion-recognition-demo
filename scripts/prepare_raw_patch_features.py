from __future__ import annotations

import _bootstrap

import argparse
from pathlib import Path

from eeg_pipeline.config import FeatureConfig, Paths
from eeg_pipeline.dataset import load_test_trials, load_train_trials
from eeg_pipeline.raw_patch import build_raw_patch_tensor, save_raw_patch_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw patch tensors for spatial-temporal models.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--patch-seconds", type=float, default=0.5)
    parser.add_argument("--patch-step-seconds", type=float, default=0.5)
    parser.add_argument("--channel-zscore", action="store_true")
    parser.add_argument("--denoise-low-hz", type=float, default=1.0)
    parser.add_argument("--denoise-high-hz", type=float, default=45.0)
    parser.add_argument("--cache-prefix", type=str, default="rawpatch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(args.root.resolve())
    cfg = FeatureConfig(
        patch_seconds=args.patch_seconds,
        patch_step_seconds=args.patch_step_seconds,
        channel_zscore=args.channel_zscore,
        denoise_low_hz=args.denoise_low_hz,
        denoise_high_hz=args.denoise_high_hz,
    )
    prefix = f"{args.cache_prefix}_"

    train_trials = load_train_trials(paths.train_root, cfg)
    train_tensors, train_meta = build_raw_patch_tensor(train_trials, cfg)
    save_raw_patch_cache(paths.artifacts_root / f"{prefix}train_rawpatch.npz", train_tensors, train_meta, cfg)

    test_trials = load_test_trials(paths.test_root, cfg)
    test_tensors, test_meta = build_raw_patch_tensor(test_trials, cfg)
    save_raw_patch_cache(paths.artifacts_root / f"{prefix}test_rawpatch.npz", test_tensors, test_meta, cfg)

    print(f"Saved train raw patch tensors: {train_tensors.shape}")
    print(f"Saved test raw patch tensors: {test_tensors.shape}")


if __name__ == "__main__":
    main()
