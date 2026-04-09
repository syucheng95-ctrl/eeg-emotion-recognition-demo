from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from eeg_pipeline.config import Paths
from train_gnn import HOLDOUT_DEFAULT


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = [
    {
        "name": "statbranch_base",
        "stat_hidden_channels": 0,
        "stat_num_layers": 2,
    },
    {
        "name": "statbranch_wide96",
        "stat_hidden_channels": 96,
        "stat_num_layers": 2,
    },
    {
        "name": "statbranch_deep3",
        "stat_hidden_channels": 64,
        "stat_num_layers": 3,
    },
    {
        "name": "statbranch_wide96_deep3",
        "stat_hidden_channels": 96,
        "stat_num_layers": 3,
    },
    {
        "name": "statbranch_wide128_deep3",
        "stat_hidden_channels": 128,
        "stat_num_layers": 3,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stronger-statistical-branch benchmarks on the current best mainline.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = Paths(root)
    rows = []
    for exp in EXPERIMENTS:
        report_name = f"{exp['name']}_report.json"
        model_name = f"{exp['name']}.pt"
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "train_multitask_dual_branch.py"),
            "--root", str(root),
            "--folds", str(args.folds),
            "--holdout-subjects", args.holdout_subjects,
            "--epochs", "100",
            "--hidden-channels", "64",
            "--dropout", "0.3",
            "--temporal-type", "gru",
            "--fusion-type", "gated",
            "--gate-mode", "vector",
            "--fusion-proj-dim", "0",
            "--gru-layers", "2",
            "--group-loss-weight", "0.02",
            "--seed", "42",
            "--standardize-sequence-input",
            "--stat-num-layers", str(exp["stat_num_layers"]),
            "--report-name", report_name,
            "--model-name", model_name,
        ]
        if int(exp["stat_hidden_channels"]) > 0:
            cmd.extend(["--stat-hidden-channels", str(exp["stat_hidden_channels"])])
        print(f"Running {exp['name']}")
        subprocess.run(cmd, cwd=root, check=True)
        report = json.loads((paths.outputs_root / report_name).read_text(encoding="utf-8"))
        holdout = report["holdout"]
        rows.append(
            {
                "experiment_name": exp["name"],
                "stat_hidden_channels": exp["stat_hidden_channels"] or 64,
                "stat_num_layers": exp["stat_num_layers"],
                "cv_mean_emotion_accuracy": report["cv_mean_emotion_accuracy"],
                "cv_mean_group_accuracy": report["cv_mean_group_accuracy"],
                "holdout_emotion_accuracy": holdout["emotion_accuracy"],
                "holdout_group_accuracy": holdout["group_accuracy"],
                "holdout_dep_accuracy": holdout["emotion_group_accuracy"].get("抑郁症患者"),
                "holdout_hc_accuracy": holdout["emotion_group_accuracy"].get("正常人"),
                "holdout_group_gap_abs": holdout["emotion_group_gap_abs"],
                "report_file": report_name,
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["holdout_emotion_accuracy", "holdout_group_gap_abs"],
        ascending=[False, True],
    )
    summary_path = paths.outputs_root / "stat_branch_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
