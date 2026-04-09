from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_pipeline.config import FeatureConfig, Paths, get_band_preset
from eeg_pipeline.dataset import load_train_trials
from eeg_pipeline.features import build_feature_matrix, load_feature_cache, save_feature_cache
from train_baseline import run_cv, run_holdout


DEFAULT_HOLDOUT = (
    "DEP1015,DEP1022,DEP1028,DEP1034,HC1010,HC1016,"
    "HC1024,HC1032,HC1038,HC1048,HC1054,HC1068"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch experiment runner for DE baselines.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=DEFAULT_HOLDOUT)
    parser.add_argument(
        "--window-seconds",
        type=str,
        default="1,2,4",
        help="Comma-separated window sizes in seconds.",
    )
    parser.add_argument(
        "--step-ratios",
        type=str,
        default="1.0,0.5",
        help="Comma-separated step/window ratios. 1.0=no overlap, 0.5=50% overlap.",
    )
    parser.add_argument(
        "--aggregations",
        type=str,
        default="mean,meanstd",
        help="Comma-separated aggregation modes.",
    )
    parser.add_argument(
        "--band-presets",
        type=str,
        default="base4,classic5",
        help="Comma-separated band preset names.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="logreg,svm,rf",
        help="Comma-separated model names to keep in the summary.",
    )
    return parser.parse_args()


def _parse_list(text: str, cast) -> list[Any]:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def _save_outputs(
    rows: list[dict[str, Any]],
    detailed_results: dict[str, Any],
    outputs_root: Path,
) -> None:
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["holdout_accuracy", "cv_mean_accuracy", "cv_mean_balanced_accuracy"],
            ascending=[False, False, False],
        )
    summary_path = outputs_root / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    detail_path = outputs_root / "experiment_details.json"
    with detail_path.open("w", encoding="utf-8") as handle:
        json.dump(detailed_results, handle, ensure_ascii=False, indent=2)

    if not summary.empty:
        best = summary.iloc[0].to_dict()
        best_model_path = outputs_root / "best_experiment_row.json"
        with best_model_path.open("w", encoding="utf-8") as handle:
            json.dump(best, handle, ensure_ascii=False, indent=2)


def _flatten_result_row(
    exp_id: str,
    cfg: FeatureConfig,
    model_name: str,
    cv_metrics: dict[str, Any],
    holdout_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "experiment_id": exp_id,
        "window_seconds": cfg.window_seconds,
        "step_seconds": cfg.step_seconds,
        "overlap_ratio": round(1.0 - (cfg.step_seconds / cfg.window_seconds), 4),
        "aggregation": cfg.aggregation,
        "band_preset": "classic5" if len(cfg.bands) == 5 else "base4",
        "feature_dim": len(cfg.bands) * 30 * (2 if cfg.aggregation == "meanstd" else 1),
        "model": model_name,
        "cv_mean_accuracy": cv_metrics["mean_accuracy"],
        "cv_mean_balanced_accuracy": cv_metrics["mean_balanced_accuracy"],
        "cv_std_accuracy": cv_metrics["std_accuracy"],
    }
    if holdout_metrics is not None:
        row["holdout_accuracy"] = holdout_metrics["accuracy"]
        row["holdout_balanced_accuracy"] = holdout_metrics["balanced_accuracy"]
        for group_name, metrics in holdout_metrics["group_metrics"].items():
            prefix = "holdout_dep" if group_name == "抑郁症患者" else "holdout_hc"
            row[f"{prefix}_accuracy"] = metrics["accuracy"]
            row[f"{prefix}_balanced_accuracy"] = metrics["balanced_accuracy"]
    return row


def main() -> None:
    args = parse_args()
    paths = Paths(args.root.resolve())
    paths.artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.outputs_root.mkdir(parents=True, exist_ok=True)

    window_seconds_list = _parse_list(args.window_seconds, float)
    step_ratios = _parse_list(args.step_ratios, float)
    aggregations = _parse_list(args.aggregations, str)
    band_presets = _parse_list(args.band_presets, str)
    selected_models = set(_parse_list(args.models, str))
    holdout_subjects = {
        item.strip() for item in args.holdout_subjects.split(",") if item.strip()
    }

    summary_path = paths.outputs_root / "experiment_summary.csv"
    detail_path = paths.outputs_root / "experiment_details.json"
    rows: list[dict[str, Any]] = []
    detailed_results: dict[str, Any] = {}
    if detail_path.exists():
        detailed_results = json.loads(detail_path.read_text(encoding="utf-8"))
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        rows = existing_summary.to_dict(orient="records")

    for window_seconds in window_seconds_list:
        for step_ratio in step_ratios:
            step_seconds = window_seconds * step_ratio
            for aggregation in aggregations:
                for band_preset in band_presets:
                    cfg = FeatureConfig(
                        window_seconds=window_seconds,
                        step_seconds=step_seconds,
                        aggregation=aggregation,
                        bands=get_band_preset(band_preset),
                    )
                    exp_id = (
                        f"w{window_seconds:g}_s{step_seconds:g}_{aggregation}_{band_preset}"
                    )
                    if exp_id in detailed_results:
                        print(f"Skipping completed experiment: {exp_id}")
                        continue
                    print(f"Running experiment: {exp_id}")
                    cache_path = paths.artifacts_root / f"{exp_id}_train_features.npz"
                    if cache_path.exists():
                        features, metadata = load_feature_cache(cache_path)
                    else:
                        train_trials = load_train_trials(paths.train_root, cfg)
                        features, metadata = build_feature_matrix(train_trials, cfg)
                        save_feature_cache(cache_path, features, metadata, cfg)

                    labels = np.array([int(item["label"]) for item in metadata], dtype=np.int64)
                    groups = np.array([str(item["subject_id"]) for item in metadata], dtype=object)
                    cv_results = run_cv(features, labels, groups, args.folds, metadata)
                    holdout_results = run_holdout(
                        features,
                        labels,
                        groups,
                        metadata,
                        holdout_subjects,
                    )
                    detailed_results[exp_id] = {
                        "config": {
                            "window_seconds": cfg.window_seconds,
                            "step_seconds": cfg.step_seconds,
                            "aggregation": cfg.aggregation,
                            "band_preset": band_preset,
                        },
                        "cross_validation": cv_results,
                        "holdout": holdout_results,
                    }
                    for model_name, cv_metrics in cv_results["models"].items():
                        if model_name not in selected_models:
                            continue
                        rows.append(
                            _flatten_result_row(
                                exp_id,
                                cfg,
                                model_name,
                                cv_metrics,
                                holdout_results["models"].get(model_name),
                            )
                        )
                    _save_outputs(rows, detailed_results, paths.outputs_root)
    _save_outputs(rows, detailed_results, paths.outputs_root)
    summary = pd.read_csv(summary_path)
    if not summary.empty:
        print("Top experiments:")
        print(summary.head(10).to_string(index=False))
    print(f"Saved summary to: {summary_path}")
    print(f"Saved details to: {detail_path}")


if __name__ == "__main__":
    main()
