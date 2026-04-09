from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from train_gnn import HOLDOUT_DEFAULT

SCRIPT_DIR = Path(__file__).resolve().parent


FAIR_EXPERIMENTS = [
    {
        "name": "baseline_logreg_meanstd",
        "kind": "baseline",
        "command": [
            str(SCRIPT_DIR / "train_baseline.py"),
            "--model", "logreg",
        ],
        "report_file": "baseline_cv_report.json",
    },
    {
        "name": "window_mlp_meanstd",
        "kind": "window",
        "command": [
            str(SCRIPT_DIR / "train_window_model.py"),
            "--epochs", "90",
            "--hidden-channels", "64",
            "--dropout", "0.3",
            "--temporal-type", "meanstd",
            "--standardize-input",
            "--report-name", "fair_window_mlp_meanstd_report.json",
            "--model-name", "fair_window_mlp_meanstd.pt",
        ],
        "report_file": "fair_window_mlp_meanstd_report.json",
        "model_file": "fair_window_mlp_meanstd.pt",
    },
    {
        "name": "window_mlp_gru",
        "kind": "window",
        "command": [
            str(SCRIPT_DIR / "train_window_model.py"),
            "--epochs", "100",
            "--hidden-channels", "64",
            "--dropout", "0.3",
            "--temporal-type", "gru",
            "--standardize-input",
            "--report-name", "fair_window_mlp_gru_report.json",
            "--model-name", "fair_window_mlp_gru.pt",
        ],
        "report_file": "fair_window_mlp_gru_report.json",
        "model_file": "fair_window_mlp_gru.pt",
    },
    {
        "name": "graph_gcn_meanstd",
        "kind": "graph",
        "command": [
            str(SCRIPT_DIR / "train_graph_model.py"),
            "--epochs", "90",
            "--hidden-channels", "64",
            "--dropout", "0.3",
            "--num-layers", "2",
            "--conv-type", "gcn",
            "--temporal-type", "meanstd",
            "--standardize-input",
            "--report-name", "fair_graph_gcn_meanstd_report.json",
            "--model-name", "fair_graph_gcn_meanstd.pt",
        ],
        "report_file": "fair_graph_gcn_meanstd_report.json",
        "model_file": "fair_graph_gcn_meanstd.pt",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed fair-comparison benchmark suite.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    return parser.parse_args()


def baseline_cache(root: Path) -> Path:
    return root / "artifacts" / "w2_s2_meanstd_classic5_train_features.npz"


def graph_cache(root: Path) -> Path:
    return root / "artifacts" / "gnn_train_graph.npz"


def parse_report(exp: dict[str, object], report: dict[str, object]) -> dict[str, object]:
    if exp["kind"] == "baseline":
        metrics = report["cross_validation"]["models"]["logreg"]
        holdout = report["holdout"]["models"]["logreg"]
        return {
            "experiment_name": exp["name"],
            "kind": exp["kind"],
            "cv_mean_accuracy": metrics["mean_accuracy"],
            "cv_std_accuracy": metrics["std_accuracy"],
            "holdout_accuracy": holdout["accuracy"],
            "holdout_dep_accuracy": holdout["group_metrics"].get("抑郁症患者", {}).get("accuracy"),
            "holdout_hc_accuracy": holdout["group_metrics"].get("正常人", {}).get("accuracy"),
            "report_file": exp["report_file"],
            "model_file": "baseline_model.pkl",
        }
    holdout = report["holdout"]
    return {
        "experiment_name": exp["name"],
        "kind": exp["kind"],
        "cv_mean_accuracy": report["cv_mean_accuracy"],
        "cv_std_accuracy": report["cv_std_accuracy"],
        "holdout_accuracy": holdout["accuracy"],
        "holdout_dep_accuracy": holdout["group_accuracy"].get("抑郁症患者"),
        "holdout_hc_accuracy": holdout["group_accuracy"].get("正常人"),
        "report_file": exp["report_file"],
        "model_file": exp.get("model_file", ""),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    outputs_root = root / "outputs_v2"
    outputs_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for exp in FAIR_EXPERIMENTS:
        cmd = [sys.executable, *exp["command"]]
        if exp["kind"] == "baseline":
            cmd.extend(
                [
                    "--root", str(root),
                    "--cache-path", str(baseline_cache(root)),
                    "--folds", str(args.folds),
                    "--holdout-subjects", args.holdout_subjects,
                ]
            )
        else:
            cmd.extend(
                [
                    "--root", str(root),
                    "--cache-path", str(graph_cache(root)),
                    "--folds", str(args.folds),
                    "--holdout-subjects", args.holdout_subjects,
                ]
            )
        print(f"Running {exp['name']}")
        subprocess.run(cmd, cwd=root, check=True)
        report = json.loads((outputs_root / exp["report_file"]).read_text(encoding="utf-8"))
        rows.append(parse_report(exp, report))

    summary = pd.DataFrame(rows).sort_values(
        ["holdout_accuracy", "cv_mean_accuracy"],
        ascending=[False, False],
    )
    summary_path = outputs_root / "fair_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
