from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_pipeline.config import Paths
from eeg_pipeline.features import load_feature_cache
from eeg_pipeline.graph_features import load_graph_cache
from eeg_pipeline.normalization import apply_window_normalizer, fit_window_normalizer
from eeg_pipeline.window_models import WindowSequenceClassifier
from train_gnn import HOLDOUT_DEFAULT


FUSION_EXPERIMENTS = [
    {
        "name": "fusion_baseline_stage2_window_mean",
        "window_temporal_type": "mean",
        "window_epochs": 90,
        "window_hidden_channels": 64,
        "window_dropout": 0.3,
        "window_standardize_input": False,
    },
    {
        "name": "fusion_baseline_window_mlp_gru",
        "window_temporal_type": "gru",
        "window_epochs": 100,
        "window_hidden_channels": 64,
        "window_dropout": 0.3,
        "window_standardize_input": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weighted-probability fusion benchmarks.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--feature-cache-path", type=Path, default=None)
    parser.add_argument("--sequence-cache-path", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-step", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_stat_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_stat_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def assert_aligned(feature_meta, sequence_meta) -> None:
    if len(feature_meta) != len(sequence_meta):
        raise ValueError("Feature cache and sequence cache have different lengths.")
    for idx, (left, right) in enumerate(zip(feature_meta, sequence_meta, strict=True)):
        left_key = (
            str(left["subject_id"]),
            str(left["trial_id"]),
            int(left["label"]),
        )
        right_key = (
            str(right["subject_id"]),
            str(right["trial_id"]),
            int(right["label"]),
        )
        if left_key != right_key:
            raise ValueError(f"Cache mismatch at index {idx}: {left_key} != {right_key}")


def build_window_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_window_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict_window_proba(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_window_model_with_labels(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    args: argparse.Namespace,
    exp: dict[str, object],
    device: torch.device,
) -> np.ndarray:
    if exp["window_standardize_input"]:
        mean, std = fit_window_normalizer(train_x)
        train_x = apply_window_normalizer(train_x, mean, std)
        valid_x = apply_window_normalizer(valid_x, mean, std)
    model = WindowSequenceClassifier(
        in_channels=train_x.shape[-1],
        hidden_channels=int(exp["window_hidden_channels"]),
        num_nodes=train_x.shape[2],
        dropout=float(exp["window_dropout"]),
        temporal_type=str(exp["window_temporal_type"]),
        temporal_heads=args.temporal_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = build_window_loader(train_x, train_y, args.batch_size, shuffle=True)
    valid_loader = build_window_loader(valid_x, valid_y, args.batch_size, shuffle=False)

    best_acc = -1.0
    best_state = None
    for _ in range(int(exp["window_epochs"])):
        run_window_epoch(model, train_loader, optimizer, criterion, device)
        probs = predict_window_proba(model, valid_loader, device)
        acc = float(accuracy_score(valid_y, (probs >= 0.5).astype(np.int64)))
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    return predict_window_proba(model, valid_loader, device)


def group_accuracy(metadata, indices, y_true, y_pred):
    result = {}
    group_names = np.array([metadata[idx]["group_name"] for idx in indices], dtype=object)
    for group_name in sorted(set(group_names.tolist())):
        mask = group_names == group_name
        result[group_name] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return result


def evaluate_fused_predictions(
    metadata,
    indices: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, object]:
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "group_accuracy": group_accuracy(metadata, indices, y_true, preds),
    }


def make_weight_grid(step: float) -> list[float]:
    count = int(round(1.0 / step))
    return [round(idx * step, 10) for idx in range(count + 1)]


def fit_baseline_proba(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray) -> np.ndarray:
    mean, std = fit_stat_scaler(train_x)
    train_x = apply_stat_scaler(train_x, mean, std)
    valid_x = apply_stat_scaler(valid_x, mean, std)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(train_x, train_y)
    return model.predict_proba(valid_x)[:, 1]


def select_best_weight(
    labels: np.ndarray,
    baseline_oof: np.ndarray,
    window_oof: np.ndarray,
    weight_grid: list[float],
) -> tuple[float, list[dict[str, float]]]:
    rows = []
    best_weight = 0.5
    best_acc = -1.0
    for weight in weight_grid:
        fused = weight * baseline_oof + (1.0 - weight) * window_oof
        preds = (fused >= 0.5).astype(np.int64)
        acc = float(accuracy_score(labels, preds))
        row = {
            "baseline_weight": weight,
            "window_weight": 1.0 - weight,
            "oof_accuracy": acc,
        }
        rows.append(row)
        if acc > best_acc:
            best_acc = acc
            best_weight = weight
    return best_weight, rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = Paths(args.root.resolve())
    feature_cache_path = args.feature_cache_path or (paths.artifacts_root / "w2_s2_meanstd_classic5_train_features.npz")
    sequence_cache_path = args.sequence_cache_path or (paths.artifacts_root / "gnn_train_graph.npz")

    features, feature_meta = load_feature_cache(feature_cache_path)
    sequences, sequence_meta = load_graph_cache(sequence_cache_path)
    assert_aligned(feature_meta, sequence_meta)
    labels = np.array([int(item["label"]) for item in feature_meta], dtype=np.int64)
    groups = np.array([str(item["subject_id"]) for item in feature_meta], dtype=object)
    holdout_subjects = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    train_mask = np.array([group not in holdout_subjects for group in groups], dtype=bool)
    valid_mask = ~train_mask
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = GroupKFold(n_splits=args.folds)
    weight_grid = make_weight_grid(args.weight_step)

    summary_rows = []
    output_payload = {
        "config": vars(args),
        "experiments": [],
    }
    output_payload["config"]["root"] = str(args.root)
    output_payload["config"]["feature_cache_path"] = (
        None if args.feature_cache_path is None else str(args.feature_cache_path)
    )
    output_payload["config"]["sequence_cache_path"] = (
        None if args.sequence_cache_path is None else str(args.sequence_cache_path)
    )

    for exp in FUSION_EXPERIMENTS:
        baseline_oof = np.zeros(len(labels), dtype=np.float32)
        window_oof = np.zeros(len(labels), dtype=np.float32)

        for fold_idx, (train_idx, fold_valid_idx) in enumerate(splitter.split(features, labels, groups), start=1):
            print(f"Running {exp['name']} fold {fold_idx}")
            baseline_oof[fold_valid_idx] = fit_baseline_proba(
                features[train_idx],
                labels[train_idx],
                features[fold_valid_idx],
            )
            window_oof[fold_valid_idx] = train_window_model_with_labels(
                sequences[train_idx],
                labels[train_idx],
                sequences[fold_valid_idx],
                labels[fold_valid_idx],
                args,
                exp,
                device,
            )

        best_weight, oof_rows = select_best_weight(labels, baseline_oof, window_oof, weight_grid)

        baseline_holdout = fit_baseline_proba(
            features[train_mask],
            labels[train_mask],
            features[valid_mask],
        )
        window_holdout = train_window_model_with_labels(
            sequences[train_mask],
            labels[train_mask],
            sequences[valid_mask],
            labels[valid_mask],
            args,
            exp,
            device,
        )
        fused_holdout = best_weight * baseline_holdout + (1.0 - best_weight) * window_holdout
        holdout_metrics = evaluate_fused_predictions(
            feature_meta,
            np.where(valid_mask)[0],
            labels[valid_mask],
            fused_holdout,
        )

        best_oof_accuracy = max(row["oof_accuracy"] for row in oof_rows)
        output_payload["experiments"].append(
            {
                "name": exp["name"],
                "window_config": exp,
                "best_baseline_weight": best_weight,
                "weight_search": oof_rows,
                "oof_accuracy": best_oof_accuracy,
                "holdout": holdout_metrics,
            }
        )
        summary_rows.append(
            {
                "experiment_name": exp["name"],
                "kind": "fusion",
                "cv_mean_accuracy": best_oof_accuracy,
                "holdout_accuracy": holdout_metrics["accuracy"],
                "holdout_dep_accuracy": holdout_metrics["group_accuracy"].get("抑郁症患者"),
                "holdout_hc_accuracy": holdout_metrics["group_accuracy"].get("正常人"),
                "best_baseline_weight": best_weight,
            }
        )

    previous_summary_path = paths.outputs_root / "fair_benchmark_summary.csv"
    if previous_summary_path.exists():
        previous_summary = pd.read_csv(previous_summary_path)
        for _, row in previous_summary.iterrows():
            summary_rows.append(
                {
                    "experiment_name": row["experiment_name"],
                    "kind": row["kind"],
                    "cv_mean_accuracy": row["cv_mean_accuracy"],
                    "holdout_accuracy": row["holdout_accuracy"],
                    "holdout_dep_accuracy": row.get("holdout_dep_accuracy"),
                    "holdout_hc_accuracy": row.get("holdout_hc_accuracy"),
                    "best_baseline_weight": np.nan,
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["holdout_accuracy", "cv_mean_accuracy"],
        ascending=[False, False],
    )
    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    report_path = paths.outputs_root / "fusion_benchmark_report.json"
    report_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = paths.outputs_root / "fusion_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved report to: {report_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
