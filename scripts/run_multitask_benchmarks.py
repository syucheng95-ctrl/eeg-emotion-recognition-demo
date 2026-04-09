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
GROUP_LOSS_WEIGHTS = [0.1, 0.3, 0.5]
FUSION_SETTINGS = [
    {
        "name_prefix": "multitask_gru_concat",
        "fusion_type": "concat",
        "temporal_type": "gru",
        "standardize_sequence_input": True,
    },
    {
        "name_prefix": "multitask_gru_gated",
        "fusion_type": "gated",
        "temporal_type": "gru",
        "standardize_sequence_input": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multitask dual-branch benchmark suite.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    return parser.parse_args()


def build_experiments() -> list[dict[str, object]]:
    experiments = []
    for setting in FUSION_SETTINGS:
        for weight in GROUP_LOSS_WEIGHTS:
            weight_tag = str(weight).replace(".", "p")
            name = f"{setting['name_prefix']}_lambda{weight_tag}"
            command = [
                str(SCRIPT_DIR / "train_multitask_dual_branch.py"),
                "--epochs", "100",
                "--hidden-channels", "64",
                "--dropout", "0.3",
                "--temporal-type", str(setting["temporal_type"]),
                "--fusion-type", str(setting["fusion_type"]),
                "--group-loss-weight", str(weight),
                "--report-name", f"{name}_report.json",
                "--model-name", f"{name}.pt",
            ]
            if setting["standardize_sequence_input"]:
                command.append("--standardize-sequence-input")
            experiments.append(
                {
                    "name": name,
                    "fusion_type": setting["fusion_type"],
                    "group_loss_weight": weight,
                    "command": command,
                    "report_file": f"{name}_report.json",
                    "model_file": f"{name}.pt",
                }
            )
    return experiments


def parse_report(exp: dict[str, object], report: dict[str, object]) -> dict[str, object]:
    holdout = report["holdout"]
    return {
        "experiment_name": exp["name"],
        "kind": "multitask_dual_branch",
        "fusion_type": exp["fusion_type"],
        "group_loss_weight": exp["group_loss_weight"],
        "cv_mean_emotion_accuracy": report["cv_mean_emotion_accuracy"],
        "cv_std_emotion_accuracy": report["cv_std_emotion_accuracy"],
        "cv_mean_group_accuracy": report["cv_mean_group_accuracy"],
        "holdout_emotion_accuracy": holdout["emotion_accuracy"],
        "holdout_group_accuracy": holdout["group_accuracy"],
        "holdout_dep_accuracy": holdout["emotion_group_accuracy"].get("抑郁症患者"),
        "holdout_hc_accuracy": holdout["emotion_group_accuracy"].get("正常人"),
        "holdout_group_gap_abs": holdout["emotion_group_gap_abs"],
        "report_file": exp["report_file"],
        "model_file": exp["model_file"],
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = Paths(root)
    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for exp in build_experiments():
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
        ["holdout_emotion_accuracy", "cv_mean_emotion_accuracy"],
        ascending=[False, False],
    )
    summary_path = paths.outputs_root / "multitask_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
