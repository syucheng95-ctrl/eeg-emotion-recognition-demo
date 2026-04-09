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


DUAL_BRANCH_EXPERIMENTS = [
    {
        "name": "dual_branch_mean_concat",
        "command": [
            str(SCRIPT_DIR / "train_dual_branch_model.py"),
            "--epochs", "90",
            "--hidden-channels", "64",
            "--dropout", "0.3",
            "--temporal-type", "mean",
            "--fusion-type", "concat",
            "--report-name", "dual_branch_mean_concat_report.json",
            "--model-name", "dual_branch_mean_concat.pt",
        ],
        "report_file": "dual_branch_mean_concat_report.json",
        "model_file": "dual_branch_mean_concat.pt",
    },
    {
        "name": "dual_branch_mean_gated",
        "command": [
            str(SCRIPT_DIR / "train_dual_branch_model.py"),
            "--epochs", "90",
            "--hidden-channels", "64",
            "--dropout", "0.3",
            "--temporal-type", "mean",
            "--fusion-type", "gated",
            "--report-name", "dual_branch_mean_gated_report.json",
            "--model-name", "dual_branch_mean_gated.pt",
        ],
        "report_file": "dual_branch_mean_gated_report.json",
        "model_file": "dual_branch_mean_gated.pt",
    },
    {
        "name": "dual_branch_gru_concat",
        "command": [
            str(SCRIPT_DIR / "train_dual_branch_model.py"),
            "--epochs", "100",
            "--hidden-channels", "64",
            "--dropout", "0.3",
            "--temporal-type", "gru",
            "--fusion-type", "concat",
            "--standardize-sequence-input",
            "--report-name", "dual_branch_gru_concat_report.json",
            "--model-name", "dual_branch_gru_concat.pt",
        ],
        "report_file": "dual_branch_gru_concat_report.json",
        "model_file": "dual_branch_gru_concat.pt",
    },
    {
        "name": "dual_branch_gru_gated",
        "command": [
            str(SCRIPT_DIR / "train_dual_branch_model.py"),
            "--epochs", "100",
            "--hidden-channels", "64",
            "--dropout", "0.3",
            "--temporal-type", "gru",
            "--fusion-type", "gated",
            "--standardize-sequence-input",
            "--report-name", "dual_branch_gru_gated_report.json",
            "--model-name", "dual_branch_gru_gated.pt",
        ],
        "report_file": "dual_branch_gru_gated_report.json",
        "model_file": "dual_branch_gru_gated.pt",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual-branch benchmark suite.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    return parser.parse_args()


def parse_report(exp: dict[str, object], report: dict[str, object]) -> dict[str, object]:
    holdout = report["holdout"]
    return {
        "experiment_name": exp["name"],
        "kind": "dual_branch",
        "cv_mean_accuracy": report["cv_mean_accuracy"],
        "cv_std_accuracy": report["cv_std_accuracy"],
        "holdout_accuracy": holdout["accuracy"],
        "holdout_dep_accuracy": holdout["group_accuracy"].get("抑郁症患者"),
        "holdout_hc_accuracy": holdout["group_accuracy"].get("正常人"),
        "report_file": exp["report_file"],
        "model_file": exp["model_file"],
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = Paths(root)
    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for exp in DUAL_BRANCH_EXPERIMENTS:
        cmd = [sys.executable, *exp["command"]]
        cmd.extend(
            [
                "--root", str(root),
                "--folds", str(args.folds),
                "--holdout-subjects", args.holdout_subjects,
            ]
        )
        print(f"Running {exp['name']}")
        subprocess.run(cmd, cwd=root, check=True)
        report = json.loads((paths.outputs_root / exp["report_file"]).read_text(encoding="utf-8"))
        rows.append(parse_report(exp, report))

    summary = pd.DataFrame(rows).sort_values(
        ["holdout_accuracy", "cv_mean_accuracy"],
        ascending=[False, False],
    )
    previous_summary_path = paths.outputs_root / "fair_benchmark_summary.csv"
    if previous_summary_path.exists():
        previous_summary = pd.read_csv(previous_summary_path)
        summary = pd.concat([summary, previous_summary], ignore_index=True, sort=False)
        summary = summary.sort_values(
            ["holdout_accuracy", "cv_mean_accuracy"],
            ascending=[False, False],
        )

    summary_path = paths.outputs_root / "dual_branch_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
