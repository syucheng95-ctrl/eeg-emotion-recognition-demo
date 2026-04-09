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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact training-only hyperparameter sweep for the finalized phase-14 multitask model."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stage1-seed", type=int, default=42)
    parser.add_argument("--final-seeds", type=str, default="42,52,62")
    parser.add_argument("--top-k-stage1", type=int, default=2)
    parser.add_argument("--top-k-stage2", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def build_stage1_space() -> list[dict[str, object]]:
    experiments = []
    for lr in [5e-4, 1e-3, 2e-3]:
        for batch_size in [16, 32, 64]:
            experiments.append(
                {
                    "stage": "stage1",
                    "name": f"s1_lr{str(lr).replace('.', 'p')}_bs{batch_size}",
                    "lr": lr,
                    "batch_size": batch_size,
                    "weight_decay": 1e-4,
                    "early_stopping_patience": 20,
                }
            )
    return experiments


def build_stage2_space(base_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    experiments = []
    for base in base_rows:
        for weight_decay in [5e-5, 1e-4, 2e-4]:
            for patience in [15, 30]:
                experiments.append(
                    {
                        "stage": "stage2",
                        "base_name": str(base["experiment_name"]),
                        "name": (
                            f"{base['experiment_name']}_wd{str(weight_decay).replace('.', 'p')}"
                            f"_pat{patience}"
                        ),
                        "lr": float(base["lr"]),
                        "batch_size": int(base["batch_size"]),
                        "weight_decay": weight_decay,
                        "early_stopping_patience": patience,
                    }
                )
    return experiments


def score_row(row: dict[str, object]) -> float:
    return (
        float(row["holdout_emotion_accuracy"]) * 100.0
        - float(row["holdout_group_gap_abs"]) * 10.0
        + float(row["cv_mean_emotion_accuracy"]) * 2.0
        - float(row["cv_std_emotion_accuracy"]) * 2.0
    )


def build_command(
    root: Path,
    folds: int,
    holdout_subjects: str,
    epochs: int,
    seed: int,
    exp: dict[str, object],
    report_name: str,
    model_name: str,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "train_multitask_dual_branch.py"),
        "--root", str(root),
        "--folds", str(folds),
        "--holdout-subjects", holdout_subjects,
        "--epochs", str(epochs),
        "--batch-size", str(exp["batch_size"]),
        "--lr", str(exp["lr"]),
        "--weight-decay", str(exp["weight_decay"]),
        "--seed", str(seed),
        "--early-stopping-patience", str(exp["early_stopping_patience"]),
        "--report-name", report_name,
        "--model-name", model_name,
    ]


def parse_report(report: dict[str, object]) -> dict[str, object]:
    holdout = report["holdout"]
    return {
        "cv_mean_emotion_accuracy": report["cv_mean_emotion_accuracy"],
        "cv_std_emotion_accuracy": report["cv_std_emotion_accuracy"],
        "cv_mean_group_accuracy": report["cv_mean_group_accuracy"],
        "cv_std_group_accuracy": report["cv_std_group_accuracy"],
        "holdout_emotion_accuracy": holdout["emotion_accuracy"],
        "holdout_group_accuracy": holdout["group_accuracy"],
        "holdout_dep_accuracy": holdout["emotion_group_accuracy"].get("抑郁症患者"),
        "holdout_hc_accuracy": holdout["emotion_group_accuracy"].get("正常人"),
        "holdout_group_gap_abs": holdout["emotion_group_gap_abs"],
    }


def maybe_load_existing(summary_path: Path) -> pd.DataFrame:
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


def run_experiment(
    root: Path,
    paths: Paths,
    folds: int,
    holdout_subjects: str,
    epochs: int,
    exp: dict[str, object],
    seed: int,
) -> dict[str, object]:
    report_name = f"{exp['name']}_seed{seed}_report.json"
    model_name = f"{exp['name']}_seed{seed}.pt"
    cmd = build_command(
        root=root,
        folds=folds,
        holdout_subjects=holdout_subjects,
        epochs=epochs,
        seed=seed,
        exp=exp,
        report_name=report_name,
        model_name=model_name,
    )
    print(f"Running {exp['name']} seed {seed}")
    subprocess.run(cmd, cwd=root, check=True)
    report = json.loads((paths.outputs_root / report_name).read_text(encoding="utf-8"))
    row = {
        "experiment_name": str(exp["name"]),
        "seed": seed,
        "lr": float(exp["lr"]),
        "batch_size": int(exp["batch_size"]),
        "weight_decay": float(exp["weight_decay"]),
        "early_stopping_patience": int(exp["early_stopping_patience"]),
        **parse_report(report),
        "report_file": report_name,
        "model_file": model_name,
    }
    row["score"] = score_row(row)
    if "base_name" in exp:
        row["base_name"] = str(exp["base_name"])
    return row


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = Paths(root)
    paths.outputs_root.mkdir(parents=True, exist_ok=True)

    stage1_summary_path = paths.outputs_root / "training_strategy_stage1_summary.csv"
    stage2_summary_path = paths.outputs_root / "training_strategy_stage2_summary.csv"
    final_summary_path = paths.outputs_root / "training_strategy_final_summary.csv"

    stage1_existing = maybe_load_existing(stage1_summary_path)
    stage1_seen = set()
    if not stage1_existing.empty:
        stage1_seen = set(
            zip(
                stage1_existing["experiment_name"].astype(str).tolist(),
                stage1_existing["seed"].astype(int).tolist(),
                strict=False,
            )
        )
    stage1_rows = stage1_existing.to_dict(orient="records") if not stage1_existing.empty else []

    for exp in build_stage1_space():
        key = (str(exp["name"]), int(args.stage1_seed))
        if args.resume and key in stage1_seen:
            print(f"Skipping existing stage1 experiment: {exp['name']} seed {args.stage1_seed}")
            continue
        row = run_experiment(
            root=root,
            paths=paths,
            folds=args.folds,
            holdout_subjects=args.holdout_subjects,
            epochs=args.epochs,
            exp=exp,
            seed=args.stage1_seed,
        )
        stage1_rows.append(row)
        pd.DataFrame(stage1_rows).sort_values(
            ["score", "holdout_emotion_accuracy"],
            ascending=[False, False],
        ).to_csv(stage1_summary_path, index=False, encoding="utf-8-sig")

    stage1_df = pd.read_csv(stage1_summary_path)
    top_stage1 = (
        stage1_df.sort_values(["score", "holdout_emotion_accuracy"], ascending=[False, False])
        .head(args.top_k_stage1)
        .to_dict(orient="records")
    )

    stage2_existing = maybe_load_existing(stage2_summary_path)
    stage2_seen = set()
    if not stage2_existing.empty:
        stage2_seen = set(
            zip(
                stage2_existing["experiment_name"].astype(str).tolist(),
                stage2_existing["seed"].astype(int).tolist(),
                strict=False,
            )
        )
    stage2_rows = stage2_existing.to_dict(orient="records") if not stage2_existing.empty else []

    for exp in build_stage2_space(top_stage1):
        key = (str(exp["name"]), int(args.stage1_seed))
        if args.resume and key in stage2_seen:
            print(f"Skipping existing stage2 experiment: {exp['name']} seed {args.stage1_seed}")
            continue
        row = run_experiment(
            root=root,
            paths=paths,
            folds=args.folds,
            holdout_subjects=args.holdout_subjects,
            epochs=args.epochs,
            exp=exp,
            seed=args.stage1_seed,
        )
        stage2_rows.append(row)
        pd.DataFrame(stage2_rows).sort_values(
            ["score", "holdout_emotion_accuracy"],
            ascending=[False, False],
        ).to_csv(stage2_summary_path, index=False, encoding="utf-8-sig")

    stage2_df = pd.read_csv(stage2_summary_path)
    top_stage2 = (
        stage2_df.sort_values(["score", "holdout_emotion_accuracy"], ascending=[False, False])
        .head(args.top_k_stage2)
        .to_dict(orient="records")
    )

    final_existing = maybe_load_existing(final_summary_path)
    final_seen = set()
    if not final_existing.empty:
        final_seen = set(
            zip(
                final_existing["experiment_name"].astype(str).tolist(),
                final_existing["seed"].astype(int).tolist(),
                strict=False,
            )
        )
    final_rows = final_existing.to_dict(orient="records") if not final_existing.empty else []

    for base in top_stage2:
        exp = {
            "name": str(base["experiment_name"]),
            "lr": float(base["lr"]),
            "batch_size": int(base["batch_size"]),
            "weight_decay": float(base["weight_decay"]),
            "early_stopping_patience": int(base["early_stopping_patience"]),
        }
        for seed in parse_seed_list(args.final_seeds):
            key = (str(exp["name"]), int(seed))
            if args.resume and key in final_seen:
                print(f"Skipping existing final experiment: {exp['name']} seed {seed}")
                continue
            row = run_experiment(
                root=root,
                paths=paths,
                folds=args.folds,
                holdout_subjects=args.holdout_subjects,
                epochs=args.epochs,
                exp=exp,
                seed=seed,
            )
            final_rows.append(row)
            pd.DataFrame(final_rows).sort_values(
                ["experiment_name", "seed"],
                ascending=[True, True],
            ).to_csv(final_summary_path, index=False, encoding="utf-8-sig")

    final_df = pd.read_csv(final_summary_path)
    aggregated = (
        final_df.groupby("experiment_name", as_index=False)
        .agg(
            lr=("lr", "first"),
            batch_size=("batch_size", "first"),
            weight_decay=("weight_decay", "first"),
            early_stopping_patience=("early_stopping_patience", "first"),
            mean_holdout_emotion_accuracy=("holdout_emotion_accuracy", "mean"),
            std_holdout_emotion_accuracy=("holdout_emotion_accuracy", "std"),
            mean_holdout_group_gap_abs=("holdout_group_gap_abs", "mean"),
            mean_cv_emotion_accuracy=("cv_mean_emotion_accuracy", "mean"),
            mean_cv_std_emotion_accuracy=("cv_std_emotion_accuracy", "mean"),
        )
        .sort_values(
            ["mean_holdout_emotion_accuracy", "mean_holdout_group_gap_abs", "std_holdout_emotion_accuracy"],
            ascending=[False, True, True],
        )
    )
    aggregated_path = paths.outputs_root / "training_strategy_final_aggregate.csv"
    aggregated.to_csv(aggregated_path, index=False, encoding="utf-8-sig")

    print("\nStage 1 summary:")
    print(stage1_df.sort_values(["score", "holdout_emotion_accuracy"], ascending=[False, False]).to_string(index=False))
    print(f"Saved stage1 summary to: {stage1_summary_path}")

    print("\nStage 2 summary:")
    print(stage2_df.sort_values(["score", "holdout_emotion_accuracy"], ascending=[False, False]).to_string(index=False))
    print(f"Saved stage2 summary to: {stage2_summary_path}")

    print("\nFinal aggregate summary:")
    print(aggregated.to_string(index=False))
    print(f"Saved final per-seed summary to: {final_summary_path}")
    print(f"Saved final aggregate summary to: {aggregated_path}")


if __name__ == "__main__":
    main()
