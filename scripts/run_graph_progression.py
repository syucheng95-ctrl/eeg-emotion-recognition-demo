from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


EXPERIMENTS = [
    {
        "name": "stage2_window_mean_s42",
        "kind": "window",
        "temporal_type": "mean",
        "hidden_channels": 64,
        "dropout": 0.30,
        "epochs": 90,
        "seed": 42,
    },
    {
        "name": "stage3_window_meanstd_s42",
        "kind": "window",
        "temporal_type": "meanstd",
        "hidden_channels": 64,
        "dropout": 0.30,
        "epochs": 90,
        "seed": 42,
    },
    {
        "name": "stage4_gcn_meanstd_s42",
        "kind": "graph",
        "conv_type": "gcn",
        "temporal_type": "meanstd",
        "num_layers": 2,
        "hidden_channels": 64,
        "dropout": 0.30,
        "epochs": 90,
        "seed": 42,
    },
    {
        "name": "stage5_cheb_meanstd_s42",
        "kind": "graph",
        "conv_type": "cheb",
        "temporal_type": "meanstd",
        "num_layers": 2,
        "hidden_channels": 64,
        "dropout": 0.30,
        "epochs": 100,
        "seed": 42,
    },
    {
        "name": "stage5_gat_meanstd_s42",
        "kind": "graph",
        "conv_type": "gat",
        "temporal_type": "meanstd",
        "num_layers": 2,
        "hidden_channels": 64,
        "dropout": 0.35,
        "epochs": 100,
        "seed": 42,
    },
    {
        "name": "stage6_gcn_gru_s42",
        "kind": "graph",
        "conv_type": "gcn",
        "temporal_type": "gru",
        "num_layers": 2,
        "hidden_channels": 64,
        "dropout": 0.30,
        "epochs": 100,
        "seed": 42,
    },
    {
        "name": "stage6_gcn_selfattn_s42",
        "kind": "graph",
        "conv_type": "gcn",
        "temporal_type": "selfattn",
        "num_layers": 2,
        "hidden_channels": 64,
        "dropout": 0.30,
        "epochs": 100,
        "seed": 42,
    },
    {
        "name": "stage7_graph_dann_s42",
        "kind": "dann",
        "hidden_channels": 64,
        "dropout": 0.30,
        "epochs": 100,
        "seed": 42,
        "lambda_domain": 0.2,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable graph progression experiments.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    outputs_root = root / "outputs_v2"
    outputs_root.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_root / "graph_progression_v2_summary.csv"
    rows = []
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict(orient="records")
    completed = {row["experiment_name"] for row in rows}
    cache_path = root / "artifacts" / "gnn_train_graph.npz"
    target_cache = root / "artifacts" / "gnn_test_graph.npz"

    for exp in EXPERIMENTS:
        if exp["name"] in completed:
            print(f"Skipping completed experiment: {exp['name']}")
            continue
        print(f"Running {exp['name']}")
        report_name = f"{exp['name']}_report.json"
        model_name = f"{exp['name']}.pt"
        if exp["kind"] == "window":
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "train_window_model.py"),
                "--root", str(root),
                "--cache-path", str(cache_path),
                "--epochs", str(exp["epochs"]),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--folds", str(args.folds),
                "--hidden-channels", str(exp["hidden_channels"]),
                "--dropout", str(exp["dropout"]),
                "--temporal-type", exp["temporal_type"],
                "--seed", str(exp["seed"]),
                "--standardize-input",
                "--report-name", report_name,
                "--model-name", model_name,
            ]
        elif exp["kind"] == "graph":
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "train_graph_model.py"),
                "--root", str(root),
                "--cache-path", str(cache_path),
                "--epochs", str(exp["epochs"]),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--folds", str(args.folds),
                "--hidden-channels", str(exp["hidden_channels"]),
                "--dropout", str(exp["dropout"]),
                "--num-layers", str(exp["num_layers"]),
                "--conv-type", exp["conv_type"],
                "--temporal-type", exp["temporal_type"],
                "--seed", str(exp["seed"]),
                "--standardize-input",
                "--report-name", report_name,
                "--model-name", model_name,
            ]
        else:
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "train_graph_dann.py"),
                "--root", str(root),
                "--source-cache", str(cache_path),
                "--target-cache", str(target_cache),
                "--epochs", str(exp["epochs"]),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--folds", str(args.folds),
                "--hidden-channels", str(exp["hidden_channels"]),
                "--dropout", str(exp["dropout"]),
                "--lambda-domain", str(exp["lambda_domain"]),
                "--seed", str(exp["seed"]),
                "--standardize-input",
                "--report-name", report_name,
                "--model-name", model_name,
            ]
        subprocess.run(cmd, cwd=root, check=True)

        report = json.loads((outputs_root / report_name).read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": exp["name"],
                "kind": exp["kind"],
                "conv_type": exp.get("conv_type", "none"),
                "temporal_type": exp.get("temporal_type", "meanstd"),
                "num_layers": exp.get("num_layers", 0),
                "hidden_channels": exp["hidden_channels"],
                "dropout": exp["dropout"],
                "epochs": exp["epochs"],
                "seed": exp["seed"],
                "cv_mean_accuracy": report["cv_mean_accuracy"],
                "cv_std_accuracy": report["cv_std_accuracy"],
                "holdout_accuracy": report["holdout"]["accuracy"],
                "holdout_dep_accuracy": report["holdout"]["group_accuracy"].get("抑郁症患者"),
                "holdout_hc_accuracy": report["holdout"]["group_accuracy"].get("正常人"),
                "report_file": report_name,
                "model_file": model_name,
            }
        )
        df = pd.DataFrame(rows).sort_values(
            ["holdout_accuracy", "cv_mean_accuracy"],
            ascending=[False, False],
        )
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    df = pd.read_csv(summary_path)
    print("Top progression experiments:")
    print(df.sort_values(["holdout_accuracy", "cv_mean_accuracy"], ascending=[False, False]).to_string(index=False))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
