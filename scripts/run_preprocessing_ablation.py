from __future__ import annotations

import _bootstrap

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from eeg_pipeline.config import FeatureConfig, Paths, get_band_preset
from eeg_pipeline.dataset import load_train_trials
from eeg_pipeline.features import build_feature_matrix, save_feature_cache
from eeg_pipeline.graph_features import build_trial_window_tensor, save_graph_cache
from train_gnn import HOLDOUT_DEFAULT

SCRIPT_DIR = Path(__file__).resolve().parent


ABLATIONS = [
    {
        "name": "zscore",
        "channel_zscore": True,
        "baseline_seconds": 0.0,
    },
    {
        "name": "baseline",
        "channel_zscore": False,
        "baseline_seconds": 0.5,
    },
    {
        "name": "zscore_baseline",
        "channel_zscore": True,
        "baseline_seconds": 0.5,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocessing ablation on baseline and sequence models.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    return parser.parse_args()


def build_cfg(ablation: dict[str, object]) -> FeatureConfig:
    return FeatureConfig(
        window_seconds=2.0,
        step_seconds=2.0,
        aggregation="meanstd",
        channel_zscore=bool(ablation["channel_zscore"]),
        baseline_seconds=float(ablation["baseline_seconds"]),
        bands=get_band_preset("classic5"),
    )


def ensure_caches(paths: Paths, cfg: FeatureConfig, prefix: str) -> tuple[Path, Path]:
    feature_cache = paths.artifacts_root / f"{prefix}_train_features.npz"
    graph_cache = paths.artifacts_root / f"{prefix}_train_graph.npz"
    if feature_cache.exists() and graph_cache.exists():
        return feature_cache, graph_cache

    train_trials = load_train_trials(paths.train_root, cfg)
    features, feature_meta = build_feature_matrix(train_trials, cfg)
    save_feature_cache(feature_cache, features, feature_meta, cfg)
    graph_tensors, graph_meta = build_trial_window_tensor(train_trials, cfg)
    save_graph_cache(graph_cache, graph_tensors, graph_meta, cfg)
    return feature_cache, graph_cache


def parse_baseline_report(report: dict[str, object]) -> dict[str, float]:
    cv = report["cross_validation"]["models"]["logreg"]
    holdout = report["holdout"]["models"]["logreg"]
    return {
        "cv_mean_accuracy": cv["mean_accuracy"],
        "cv_std_accuracy": cv["std_accuracy"],
        "holdout_accuracy": holdout["accuracy"],
        "holdout_dep_accuracy": holdout["group_metrics"]["抑郁症患者"]["accuracy"],
        "holdout_hc_accuracy": holdout["group_metrics"]["正常人"]["accuracy"],
    }


def parse_sequence_report(report: dict[str, object]) -> dict[str, float]:
    holdout = report["holdout"]
    return {
        "cv_mean_accuracy": report["cv_mean_accuracy"],
        "cv_std_accuracy": report["cv_std_accuracy"],
        "holdout_accuracy": holdout["accuracy"],
        "holdout_dep_accuracy": holdout["group_accuracy"]["抑郁症患者"],
        "holdout_hc_accuracy": holdout["group_accuracy"]["正常人"],
    }


def run_command(cmd: list[str], root: Path) -> None:
    subprocess.run(cmd, cwd=root, check=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = Paths(root)
    paths.outputs_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    fair_rows = {row["experiment_name"]: row for row in load_csv_rows(paths.outputs_root / "fair_benchmark_summary.csv")}
    stage_rows = {row["experiment_name"]: row for row in load_csv_rows(paths.outputs_root / "graph_progression_v2_summary.csv")}

    raw_baseline = fair_rows["baseline_logreg_meanstd"]
    summary_rows.append(
        {
            "preprocessing": "raw",
            "model_family": "baseline",
            "cv_mean_accuracy": raw_baseline["cv_mean_accuracy"],
            "cv_std_accuracy": raw_baseline["cv_std_accuracy"],
            "holdout_accuracy": raw_baseline["holdout_accuracy"],
            "holdout_dep_accuracy": raw_baseline["holdout_dep_accuracy"],
            "holdout_hc_accuracy": raw_baseline["holdout_hc_accuracy"],
            "report_file": raw_baseline["report_file"],
        }
    )

    raw_seq = stage_rows["stage2_window_mean_s42"]
    summary_rows.append(
        {
            "preprocessing": "raw",
            "model_family": "sequence",
            "cv_mean_accuracy": raw_seq["cv_mean_accuracy"],
            "cv_std_accuracy": raw_seq["cv_std_accuracy"],
            "holdout_accuracy": raw_seq["holdout_accuracy"],
            "holdout_dep_accuracy": raw_seq["holdout_dep_accuracy"],
            "holdout_hc_accuracy": raw_seq["holdout_hc_accuracy"],
            "report_file": raw_seq["report_file"],
        }
    )

    for ablation in ABLATIONS:
        cfg = build_cfg(ablation)
        prefix = f"prep_{ablation['name']}"
        feature_cache, graph_cache = ensure_caches(paths, cfg, prefix)

        baseline_report_name = f"{prefix}_baseline_report.json"
        run_command(
            [
                sys.executable,
                str(SCRIPT_DIR / "train_baseline.py"),
                "--root", str(root),
                "--cache-path", str(feature_cache),
                "--folds", str(args.folds),
                "--holdout-subjects", args.holdout_subjects,
                "--model", "logreg",
            ],
            root,
        )
        default_baseline_report = paths.outputs_root / "baseline_cv_report.json"
        baseline_target = paths.outputs_root / baseline_report_name
        baseline_target.write_text(default_baseline_report.read_text(encoding="utf-8"), encoding="utf-8")
        baseline_metrics = parse_baseline_report(json.loads(baseline_target.read_text(encoding="utf-8")))
        summary_rows.append(
            {
                "preprocessing": ablation["name"],
                "model_family": "baseline",
                **baseline_metrics,
                "report_file": baseline_report_name,
            }
        )

        sequence_report_name = f"{prefix}_stage2_window_mean_report.json"
        run_command(
            [
                sys.executable,
                str(SCRIPT_DIR / "train_window_model.py"),
                "--root", str(root),
                "--cache-path", str(graph_cache),
                "--folds", str(args.folds),
                "--holdout-subjects", args.holdout_subjects,
                "--epochs", "90",
                "--hidden-channels", "64",
                "--dropout", "0.3",
                "--temporal-type", "mean",
                "--seed", "42",
                "--standardize-input",
                "--report-name", sequence_report_name,
                "--model-name", f"{prefix}_stage2_window_mean.pt",
            ],
            root,
        )
        sequence_metrics = parse_sequence_report(
            json.loads((paths.outputs_root / sequence_report_name).read_text(encoding="utf-8"))
        )
        summary_rows.append(
            {
                "preprocessing": ablation["name"],
                "model_family": "sequence",
                **sequence_metrics,
                "report_file": sequence_report_name,
            }
        )

    for row in summary_rows:
        dep = float(row["holdout_dep_accuracy"])
        hc = float(row["holdout_hc_accuracy"])
        row["holdout_group_gap_abs"] = abs(dep - hc)

    summary_path = paths.outputs_root / "preprocessing_ablation_summary.csv"
    fieldnames = [
        "preprocessing",
        "model_family",
        "cv_mean_accuracy",
        "cv_std_accuracy",
        "holdout_accuracy",
        "holdout_dep_accuracy",
        "holdout_hc_accuracy",
        "holdout_group_gap_abs",
        "report_file",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary to: {summary_path}")
    for row in summary_rows:
        print(
            f"{row['model_family']} | {row['preprocessing']} | "
            f"holdout={float(row['holdout_accuracy']):.4f} | "
            f"dep={float(row['holdout_dep_accuracy']):.4f} | "
            f"hc={float(row['holdout_hc_accuracy']):.4f} | "
            f"gap={float(row['holdout_group_gap_abs']):.4f}"
        )


if __name__ == "__main__":
    main()
