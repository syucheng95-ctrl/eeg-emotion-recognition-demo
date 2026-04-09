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
        "name": "gcn_l2_h32_mean",
        "epochs": 80,
        "hidden_channels": 32,
        "dropout": 0.30,
        "num_layers": 2,
        "window_pool": "mean",
    },
    {
        "name": "gcn_l2_h64_mean",
        "epochs": 80,
        "hidden_channels": 64,
        "dropout": 0.30,
        "num_layers": 2,
        "window_pool": "mean",
    },
    {
        "name": "gcn_l3_h64_mean",
        "epochs": 100,
        "hidden_channels": 64,
        "dropout": 0.35,
        "num_layers": 3,
        "window_pool": "mean",
    },
    {
        "name": "gcn_l2_h64_attn",
        "epochs": 100,
        "hidden_channels": 64,
        "dropout": 0.35,
        "num_layers": 2,
        "window_pool": "attention",
    },
    {
        "name": "gcn_l3_h64_attn",
        "epochs": 120,
        "hidden_channels": 64,
        "dropout": 0.35,
        "num_layers": 3,
        "window_pool": "attention",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable GNN experiment suite.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-prefix", type=str, default="gnn")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    outputs_root = root / "outputs_v2"
    outputs_root.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_root / "gnn_experiment_summary.csv"
    rows = []
    if summary_path.exists():
        rows = pd.read_csv(summary_path).to_dict(orient="records")
    completed = {row["experiment_name"] for row in rows}

    cache_path = root / "artifacts" / f"{args.cache_prefix}_train_graph.npz"
    for exp in EXPERIMENTS:
        if exp["name"] in completed:
            print(f"Skipping completed experiment: {exp['name']}")
            continue
        report_name = f"{exp['name']}_report.json"
        model_name = f"{exp['name']}.pt"
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "train_gnn.py"),
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
            "--window-pool", exp["window_pool"],
            "--seed", str(args.seed),
            "--report-name", report_name,
            "--model-name", model_name,
        ]
        print(f"Running {exp['name']}")
        subprocess.run(cmd, cwd=root, check=True)

        report = json.loads((outputs_root / report_name).read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": exp["name"],
                "epochs": exp["epochs"],
                "hidden_channels": exp["hidden_channels"],
                "dropout": exp["dropout"],
                "num_layers": exp["num_layers"],
                "window_pool": exp["window_pool"],
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
    print("Top GNN experiments:")
    print(df.sort_values(["holdout_accuracy", "cv_mean_accuracy"], ascending=[False, False]).to_string(index=False))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
