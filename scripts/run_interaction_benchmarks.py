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
        "name": "interaction_base_none",
        "interaction_mode": "none",
    },
    {
        "name": "interaction_seq_gate",
        "interaction_mode": "seq_gate",
    },
    {
        "name": "interaction_film",
        "interaction_mode": "film",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run branch-interaction benchmarks on the current best multitask mainline.")
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
            "--stat-hidden-channels", "96",
            "--stat-num-layers", "2",
            "--interaction-mode", exp["interaction_mode"],
            "--report-name", report_name,
            "--model-name", model_name,
        ]
        print(f"Running {exp['name']}")
        subprocess.run(cmd, cwd=root, check=True)
        report = json.loads((paths.outputs_root / report_name).read_text(encoding="utf-8"))
        holdout = report["holdout"]
        rows.append(
            {
                "experiment_name": exp["name"],
                "interaction_mode": exp["interaction_mode"],
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
    summary_path = paths.outputs_root / "interaction_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
