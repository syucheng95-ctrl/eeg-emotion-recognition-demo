from __future__ import annotations

import _bootstrap

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from eeg_pipeline.config import FeatureConfig, Paths
from eeg_pipeline.dataset import load_train_trials
from eeg_pipeline.raw_patch import build_raw_patch_tensor, save_raw_patch_cache
from train_gnn import HOLDOUT_DEFAULT

SCRIPT_DIR = Path(__file__).resolve().parent


RAW_EXPERIMENTS = [
    {
        "name": "rawpatch_a",
        "variant": "a",
        "epochs": 70,
        "batch_size": 16,
        "channel_hidden": 16,
        "graph_hidden": 64,
        "dropout": 0.3,
    },
    {
        "name": "rawpatch_b",
        "variant": "b",
        "epochs": 80,
        "batch_size": 16,
        "channel_hidden": 16,
        "graph_hidden": 64,
        "dropout": 0.3,
    },
    {
        "name": "rawpatch_c",
        "variant": "c",
        "epochs": 90,
        "batch_size": 16,
        "channel_hidden": 16,
        "graph_hidden": 64,
        "dropout": 0.3,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw patch A/B/C benchmarks and compare against existing baselines.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_raw_cache(paths: Paths) -> Path:
    cache_path = paths.artifacts_root / "rawpatch_train_rawpatch.npz"
    if cache_path.exists():
        return cache_path
    cfg = FeatureConfig(
        patch_seconds=0.5,
        patch_step_seconds=0.5,
        denoise_low_hz=1.0,
        denoise_high_hz=45.0,
    )
    train_trials = load_train_trials(paths.train_root, cfg)
    tensors, metadata = build_raw_patch_tensor(train_trials, cfg)
    save_raw_patch_cache(cache_path, tensors, metadata, cfg)
    return cache_path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = Paths(root)
    outputs_root = paths.outputs_root
    outputs_root.mkdir(parents=True, exist_ok=True)
    raw_cache = ensure_raw_cache(paths)

    rows: list[dict[str, object]] = []
    compare_sources = [
        outputs_root / "fair_benchmark_summary.csv",
        outputs_root / "graph_progression_v2_summary.csv",
        outputs_root / "preprocessing_ablation_summary.csv",
    ]
    for source in compare_sources:
        if not source.exists():
            continue
        for row in load_csv_rows(source):
            experiment_name = row.get("experiment_name")
            if not experiment_name:
                model_family = row.get("model_family", "existing")
                preprocessing = row.get("preprocessing", "unknown")
                experiment_name = f"{model_family}_{preprocessing}"
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "kind": row.get("kind", row.get("model_family", "existing")),
                    "cv_mean_accuracy": float(row["cv_mean_accuracy"]),
                    "cv_std_accuracy": float(row["cv_std_accuracy"]),
                    "holdout_accuracy": float(row["holdout_accuracy"]),
                    "holdout_dep_accuracy": float(row["holdout_dep_accuracy"]),
                    "holdout_hc_accuracy": float(row["holdout_hc_accuracy"]),
                    "report_file": row["report_file"],
                }
            )

    for exp in RAW_EXPERIMENTS:
        report_name = f"{exp['name']}_report.json"
        model_name = f"{exp['name']}.pt"
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "train_raw_patch_model.py"),
            "--root", str(root),
            "--cache-path", str(raw_cache),
            "--variant", exp["variant"],
            "--epochs", str(exp["epochs"]),
            "--batch-size", str(exp["batch_size"]),
            "--folds", str(args.folds),
            "--channel-hidden", str(exp["channel_hidden"]),
            "--graph-hidden", str(exp["graph_hidden"]),
            "--dropout", str(exp["dropout"]),
            "--holdout-subjects", args.holdout_subjects,
            "--report-name", report_name,
            "--model-name", model_name,
        ]
        print(f"Running {exp['name']}")
        subprocess.run(cmd, cwd=root, check=True)
        report = json.loads((outputs_root / report_name).read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": exp["name"],
                "kind": "rawpatch",
                "cv_mean_accuracy": report["cv_mean_accuracy"],
                "cv_std_accuracy": report["cv_std_accuracy"],
                "holdout_accuracy": report["holdout"]["accuracy"],
                "holdout_dep_accuracy": report["holdout"]["group_accuracy"]["抑郁症患者"],
                "holdout_hc_accuracy": report["holdout"]["group_accuracy"]["正常人"],
                "report_file": report_name,
            }
        )

    deduped = {}
    for row in rows:
        deduped[row["experiment_name"]] = row
    final_rows = sorted(
        deduped.values(),
        key=lambda item: (item["holdout_accuracy"], item["cv_mean_accuracy"]),
        reverse=True,
    )

    summary_path = outputs_root / "raw_patch_benchmark_summary.csv"
    fieldnames = [
        "experiment_name",
        "kind",
        "cv_mean_accuracy",
        "cv_std_accuracy",
        "holdout_accuracy",
        "holdout_dep_accuracy",
        "holdout_hc_accuracy",
        "report_file",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Saved summary to: {summary_path}")
    for row in final_rows[:12]:
        print(
            f"{row['experiment_name']} | {row['kind']} | "
            f"holdout={row['holdout_accuracy']:.4f} | cv={row['cv_mean_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
