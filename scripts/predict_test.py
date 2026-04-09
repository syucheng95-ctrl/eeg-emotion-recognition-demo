from __future__ import annotations

import _bootstrap

import argparse
from pathlib import Path
import pickle

import pandas as pd

from eeg_pipeline.config import Paths
from eeg_pipeline.features import load_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict public test labels and export Excel.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root path.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional trained model path. Defaults to outputs_v2/baseline_model.pkl",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Optional test feature cache path. Defaults to artifacts/test_features.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(args.root.resolve())
    model_path = args.model_path or (paths.outputs_root / "baseline_model.pkl")
    cache_path = args.cache_path or (paths.artifacts_root / "test_features.npz")

    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    features, metadata = load_feature_cache(cache_path)
    preds = model.predict(features)

    rows = []
    for meta, pred in zip(metadata, preds, strict=True):
        rows.append(
            {
                "user_id": meta["subject_id"],
                "trial_id": int(meta["trial_id"]),
                "Emotion_label": int(pred),
            }
        )

    submission = pd.DataFrame(rows).sort_values(["user_id", "trial_id"]).reset_index(drop=True)
    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    out_path = paths.outputs_root / "submission_public_test.xlsx"
    submission.to_excel(out_path, index=False)
    print(f"Saved submission to: {out_path}")


if __name__ == "__main__":
    main()
