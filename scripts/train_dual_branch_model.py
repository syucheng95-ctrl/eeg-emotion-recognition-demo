from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_pipeline.config import Paths
from eeg_pipeline.dual_branch_models import DualBranchClassifier
from eeg_pipeline.features import load_feature_cache
from eeg_pipeline.graph_features import load_graph_cache
from eeg_pipeline.normalization import apply_window_normalizer, fit_window_normalizer
from train_gnn import HOLDOUT_DEFAULT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight dual-branch model with subject-level validation.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--feature-cache-path", type=Path, default=None)
    parser.add_argument("--sequence-cache-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temporal-type", type=str, default="mean", choices=["mean", "attention", "meanstd", "gru", "selfattn"])
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--fusion-type", type=str, default="concat", choices=["concat", "gated"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize-sequence-input", action="store_true")
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--report-name", type=str, default="dual_branch_model_report.json")
    parser.add_argument("--model-name", type=str, default="dual_branch_model.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(
    stat_x: np.ndarray,
    seq_x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(stat_x).float(),
        torch.from_numpy(seq_x).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fit_stat_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_stat_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def group_accuracy(metadata, indices, preds, labels):
    result = {}
    group_names = np.array([metadata[idx]["group_name"] for idx in indices], dtype=object)
    for group_name in sorted(set(group_names.tolist())):
        mask = group_names == group_name
        result[group_name] = float(accuracy_score(labels[mask], preds[mask]))
    return result


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for stat_xb, seq_xb, yb in loader:
        stat_xb = stat_xb.to(device)
        seq_xb = seq_xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(stat_xb, seq_xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    for stat_xb, seq_xb, yb in loader:
        stat_xb = stat_xb.to(device)
        seq_xb = seq_xb.to(device)
        logits = model(stat_xb, seq_xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.numpy().tolist())
    acc = float(accuracy_score(all_true, all_pred))
    cm = confusion_matrix(all_true, all_pred).tolist()
    return acc, cm, np.array(all_pred, dtype=np.int64)


def maybe_standardize_inputs(
    stat_train: np.ndarray,
    stat_valid: np.ndarray,
    seq_train: np.ndarray,
    seq_valid: np.ndarray,
    args: argparse.Namespace,
):
    stat_mean, stat_std = fit_stat_scaler(stat_train)
    stat_train = apply_stat_scaler(stat_train, stat_mean, stat_std)
    stat_valid = apply_stat_scaler(stat_valid, stat_mean, stat_std)
    stats = {
        "stat_mean": stat_mean,
        "stat_std": stat_std,
    }

    if args.standardize_sequence_input:
        seq_mean, seq_std = fit_window_normalizer(seq_train)
        seq_train = apply_window_normalizer(seq_train, seq_mean, seq_std)
        seq_valid = apply_window_normalizer(seq_valid, seq_mean, seq_std)
        stats["seq_mean"] = seq_mean
        stats["seq_std"] = seq_std
    return stat_train, stat_valid, seq_train, seq_valid, stats


def train_one_split(
    stat_train: np.ndarray,
    y_train: np.ndarray,
    seq_train: np.ndarray,
    stat_valid: np.ndarray,
    y_valid: np.ndarray,
    seq_valid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
):
    stat_train, stat_valid, seq_train, seq_valid, stats = maybe_standardize_inputs(
        stat_train,
        stat_valid,
        seq_train,
        seq_valid,
        args,
    )
    model = DualBranchClassifier(
        stat_in_features=stat_train.shape[-1],
        seq_in_channels=seq_train.shape[-1],
        seq_num_nodes=seq_train.shape[2],
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        temporal_type=args.temporal_type,
        temporal_heads=args.temporal_heads,
        fusion_type=args.fusion_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = build_loader(stat_train, seq_train, y_train, args.batch_size, shuffle=True)
    valid_loader = build_loader(stat_valid, seq_valid, y_valid, args.batch_size, shuffle=False)

    best_acc = -1.0
    best_state = None
    for _ in range(args.epochs):
        run_epoch(model, train_loader, optimizer, criterion, device)
        acc, _, _ = evaluate(model, valid_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    acc, cm, preds = evaluate(model, valid_loader, device)
    return model, acc, cm, preds, stats


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splitter = GroupKFold(n_splits=args.folds)
    fold_results = []
    all_true, all_pred = [], []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, labels, groups), start=1):
        _, acc, cm, preds, _ = train_one_split(
            features[train_idx],
            labels[train_idx],
            sequences[train_idx],
            features[valid_idx],
            labels[valid_idx],
            sequences[valid_idx],
            args,
            device,
        )
        fold_results.append(
            {
                "fold": fold_idx,
                "accuracy": acc,
                "subjects": sorted(set(groups[valid_idx].tolist())),
                "confusion_matrix": cm,
                "group_accuracy": group_accuracy(feature_meta, valid_idx, preds, labels[valid_idx]),
            }
        )
        all_true.extend(labels[valid_idx].tolist())
        all_pred.extend(preds.tolist())
        print(f"Fold {fold_idx}: acc={acc:.4f}")

    holdout_subjects = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    train_mask = np.array([group not in holdout_subjects for group in groups], dtype=bool)
    valid_mask = ~train_mask
    _, holdout_acc, holdout_cm, holdout_preds, _ = train_one_split(
        features[train_mask],
        labels[train_mask],
        sequences[train_mask],
        features[valid_mask],
        labels[valid_mask],
        sequences[valid_mask],
        args,
        device,
    )
    print(f"Holdout acc={holdout_acc:.4f}")

    final_features = features
    final_sequences = sequences
    final_stats = {}
    stat_mean, stat_std = fit_stat_scaler(final_features)
    final_features = apply_stat_scaler(final_features, stat_mean, stat_std)
    final_stats["stat_mean"] = stat_mean
    final_stats["stat_std"] = stat_std
    if args.standardize_sequence_input:
        seq_mean, seq_std = fit_window_normalizer(final_sequences)
        final_sequences = apply_window_normalizer(final_sequences, seq_mean, seq_std)
        final_stats["seq_mean"] = seq_mean
        final_stats["seq_std"] = seq_std

    final_model = DualBranchClassifier(
        stat_in_features=final_features.shape[-1],
        seq_in_channels=final_sequences.shape[-1],
        seq_num_nodes=final_sequences.shape[2],
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        temporal_type=args.temporal_type,
        temporal_heads=args.temporal_heads,
        fusion_type=args.fusion_type,
    ).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    final_loader = build_loader(final_features, final_sequences, labels, args.batch_size, shuffle=True)
    for _ in range(args.epochs):
        run_epoch(final_model, final_loader, optimizer, criterion, device)

    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    report = {
        "config": vars(args),
        "folds": fold_results,
        "cv_mean_accuracy": float(np.mean([item["accuracy"] for item in fold_results])),
        "cv_std_accuracy": float(np.std([item["accuracy"] for item in fold_results])),
        "overall_confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
        "holdout": {
            "subjects": sorted(holdout_subjects),
            "accuracy": holdout_acc,
            "confusion_matrix": holdout_cm,
            "group_accuracy": group_accuracy(feature_meta, np.where(valid_mask)[0], holdout_preds, labels[valid_mask]),
        },
    }
    report["config"]["root"] = str(args.root)
    report["config"]["feature_cache_path"] = str(feature_cache_path)
    report["config"]["sequence_cache_path"] = str(sequence_cache_path)
    report_path = paths.outputs_root / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    model_payload = {
        "state_dict": final_model.state_dict(),
        "stat_in_features": final_features.shape[-1],
        "seq_in_channels": final_sequences.shape[-1],
        "seq_num_nodes": final_sequences.shape[2],
        "hidden_channels": args.hidden_channels,
        "dropout": args.dropout,
        "temporal_type": args.temporal_type,
        "temporal_heads": args.temporal_heads,
        "fusion_type": args.fusion_type,
        **final_stats,
    }
    model_path = paths.outputs_root / args.model_name
    torch.save(model_payload, model_path)
    print(f"Saved report to: {report_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
