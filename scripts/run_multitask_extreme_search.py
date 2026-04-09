from __future__ import annotations

import _bootstrap

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from eeg_pipeline.config import Paths
from train_gnn import HOLDOUT_DEFAULT


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged extreme search for multitask gated dual-branch models.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--phase1-seed", type=int, default=42)
    parser.add_argument("--phase2-seeds", type=str, default="42,52,62")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def build_search_space() -> list[dict[str, object]]:
    search_space = []
    lambda_values = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    gate_modes = ["vector", "scalar", "dual"]
    fusion_proj_dims = [0, 32]
    hidden_channels = [48, 64, 80]
    gru_layers = [1, 2]
    dropouts = [0.2, 0.3, 0.4]

    for group_loss_weight, gate_mode, fusion_proj_dim, hidden, gru_layer, dropout in itertools.product(
        lambda_values,
        gate_modes,
        fusion_proj_dims,
        hidden_channels,
        gru_layers,
        dropouts,
    ):
        lambda_tag = str(group_loss_weight).replace(".", "p")
        proj_tag = "none" if fusion_proj_dim == 0 else str(fusion_proj_dim)
        dropout_tag = str(dropout).replace(".", "p")
        name = (
            f"extreme_mt_lambda{lambda_tag}_gate{gate_mode}_proj{proj_tag}"
            f"_h{hidden}_gru{gru_layer}_d{dropout_tag}"
        )
        search_space.append(
            {
                "name": name,
                "group_loss_weight": group_loss_weight,
                "gate_mode": gate_mode,
                "fusion_proj_dim": fusion_proj_dim,
                "hidden_channels": hidden,
                "gru_layers": gru_layer,
                "dropout": dropout,
            }
        )
    return search_space


def score_row(row: dict[str, object]) -> float:
    return (
        float(row["holdout_emotion_accuracy"]) * 100.0
        - float(row["holdout_group_gap_abs"]) * 10.0
        + float(row["cv_mean_emotion_accuracy"]) * 2.0
    )


def build_command(
    root: Path,
    folds: int,
    holdout_subjects: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
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
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--hidden-channels", str(exp["hidden_channels"]),
        "--dropout", str(exp["dropout"]),
        "--temporal-type", "gru",
        "--fusion-type", "gated",
        "--gate-mode", str(exp["gate_mode"]),
        "--fusion-proj-dim", str(exp["fusion_proj_dim"]),
        "--gru-layers", str(exp["gru_layers"]),
        "--group-loss-weight", str(exp["group_loss_weight"]),
        "--seed", str(seed),
        "--standardize-sequence-input",
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


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = Paths(root)
    paths.outputs_root.mkdir(parents=True, exist_ok=True)

    phase1_summary_path = paths.outputs_root / "multitask_extreme_phase1_summary.csv"
    phase2_summary_path = paths.outputs_root / "multitask_extreme_phase2_summary.csv"
    phase1_rows = maybe_load_existing(phase1_summary_path)
    phase2_rows = maybe_load_existing(phase2_summary_path)

    seen_phase1 = set()
    if not phase1_rows.empty:
        seen_phase1 = set(phase1_rows["experiment_name"].astype(str).tolist())

    rows = phase1_rows.to_dict(orient="records") if not phase1_rows.empty else []
    for exp in build_search_space():
        exp_name = str(exp["name"])
        if args.resume and exp_name in seen_phase1:
            print(f"Skipping existing phase1 experiment: {exp_name}")
            continue
        report_name = f"{exp_name}_s{args.phase1_seed}_report.json"
        model_name = f"{exp_name}_s{args.phase1_seed}.pt"
        cmd = build_command(
            root=root,
            folds=args.folds,
            holdout_subjects=args.holdout_subjects,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.phase1_seed,
            exp=exp,
            report_name=report_name,
            model_name=model_name,
        )
        print(f"Phase1 running {exp_name}")
        subprocess.run(cmd, cwd=root, check=True)
        report = json.loads((paths.outputs_root / report_name).read_text(encoding="utf-8"))
        row = {
            "experiment_name": exp_name,
            "seed": args.phase1_seed,
            **exp,
            **parse_report(report),
        }
        row["score"] = score_row(row)
        row["report_file"] = report_name
        row["model_file"] = model_name
        rows.append(row)
        phase1_df = pd.DataFrame(rows).sort_values(["score", "holdout_emotion_accuracy"], ascending=[False, False])
        phase1_df.to_csv(phase1_summary_path, index=False, encoding="utf-8-sig")

    phase1_df = pd.read_csv(phase1_summary_path)
    phase1_ranked = phase1_df.sort_values(["score", "holdout_emotion_accuracy"], ascending=[False, False])
    top_phase1 = phase1_ranked.head(args.top_k)

    phase2_seed_list = parse_seed_list(args.phase2_seeds)
    phase2_existing = set()
    if not phase2_rows.empty:
        phase2_existing = set(
            zip(
                phase2_rows["base_experiment_name"].astype(str).tolist(),
                phase2_rows["seed"].astype(int).tolist(),
                strict=False,
            )
        )
    phase2_records = phase2_rows.to_dict(orient="records") if not phase2_rows.empty else []

    for _, exp_row in top_phase1.iterrows():
        exp = {
            "name": exp_row["experiment_name"],
            "group_loss_weight": float(exp_row["group_loss_weight"]),
            "gate_mode": str(exp_row["gate_mode"]),
            "fusion_proj_dim": int(exp_row["fusion_proj_dim"]),
            "hidden_channels": int(exp_row["hidden_channels"]),
            "gru_layers": int(exp_row["gru_layers"]),
            "dropout": float(exp_row["dropout"]),
        }
        for seed in phase2_seed_list:
            key = (str(exp["name"]), int(seed))
            if args.resume and key in phase2_existing:
                print(f"Skipping existing phase2 experiment: {exp['name']} seed {seed}")
                continue
            run_name = f"{exp['name']}_seed{seed}"
            report_name = f"{run_name}_report.json"
            model_name = f"{run_name}.pt"
            cmd = build_command(
                root=root,
                folds=args.folds,
                holdout_subjects=args.holdout_subjects,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=seed,
                exp=exp,
                report_name=report_name,
                model_name=model_name,
            )
            print(f"Phase2 running {exp['name']} seed {seed}")
            subprocess.run(cmd, cwd=root, check=True)
            report = json.loads((paths.outputs_root / report_name).read_text(encoding="utf-8"))
            row = {
                "base_experiment_name": exp["name"],
                "seed": seed,
                **exp,
                **parse_report(report),
            }
            row["score"] = score_row(row)
            row["report_file"] = report_name
            row["model_file"] = model_name
            phase2_records.append(row)
            pd.DataFrame(phase2_records).to_csv(phase2_summary_path, index=False, encoding="utf-8-sig")

    phase2_df = pd.read_csv(phase2_summary_path) if phase2_summary_path.exists() else pd.DataFrame()
    if not phase2_df.empty:
        aggregate = (
            phase2_df.groupby("base_experiment_name", as_index=False)
            .agg(
                group_loss_weight=("group_loss_weight", "first"),
                gate_mode=("gate_mode", "first"),
                fusion_proj_dim=("fusion_proj_dim", "first"),
                hidden_channels=("hidden_channels", "first"),
                gru_layers=("gru_layers", "first"),
                dropout=("dropout", "first"),
                mean_holdout_emotion_accuracy=("holdout_emotion_accuracy", "mean"),
                std_holdout_emotion_accuracy=("holdout_emotion_accuracy", "std"),
                mean_holdout_group_gap_abs=("holdout_group_gap_abs", "mean"),
                mean_cv_emotion_accuracy=("cv_mean_emotion_accuracy", "mean"),
                mean_score=("score", "mean"),
            )
            .sort_values(["mean_score", "mean_holdout_emotion_accuracy"], ascending=[False, False])
        )
        aggregate_path = paths.outputs_root / "multitask_extreme_phase2_aggregate.csv"
        aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
        print(aggregate.to_string(index=False))
        print(f"Saved aggregate summary to: {aggregate_path}")

    print(f"Saved phase1 summary to: {phase1_summary_path}")
    print(f"Saved phase2 summary to: {phase2_summary_path}")


if __name__ == "__main__":
    main()
