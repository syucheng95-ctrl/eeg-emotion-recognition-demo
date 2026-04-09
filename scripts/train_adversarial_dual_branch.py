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

from eeg_pipeline.adversarial_models import AdversarialDualBranchClassifier
from eeg_pipeline.config import Paths
from eeg_pipeline.features import load_feature_cache
from eeg_pipeline.graph_features import load_graph_cache
from eeg_pipeline.normalization import apply_window_normalizer, fit_window_normalizer
from train_gnn import HOLDOUT_DEFAULT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adversarial dual-branch model with gradient-reversal group head.")
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
    parser.add_argument("--temporal-type", type=str, default="gru", choices=["mean", "attention", "meanstd", "gru", "selfattn"])
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--fusion-type", type=str, default="gated", choices=["concat", "gated"])
    parser.add_argument("--gate-mode", type=str, default="vector", choices=["vector", "scalar", "dual"])
    parser.add_argument("--fusion-proj-dim", type=int, default=0)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--adversarial-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize-sequence-input", action="store_true")
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--report-name", type=str, default="adversarial_dual_branch_report.json")
    parser.add_argument("--model-name", type=str, default="adversarial_dual_branch.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(stat_x, seq_x, emotion_y, group_y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(stat_x).float(),
        torch.from_numpy(seq_x).float(),
        torch.from_numpy(emotion_y).long(),
        torch.from_numpy(group_y).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fit_stat_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_stat_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def safe_accuracy(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(accuracy_score(y_true, y_pred))


def emotion_group_accuracy(metadata, indices, preds, labels):
    result = {}
    group_names = np.array([metadata[idx]["group_name"] for idx in indices], dtype=object)
    for group_name in sorted(set(group_names.tolist())):
        mask = group_names == group_name
        result[group_name] = safe_accuracy(labels[mask], preds[mask])
    return result


def run_epoch(model, loader, optimizer, emotion_criterion, group_criterion, adversarial_weight, device):
    model.train()
    total_loss = 0.0
    for stat_xb, seq_xb, emotion_yb, group_yb in loader:
        stat_xb = stat_xb.to(device)
        seq_xb = seq_xb.to(device)
        emotion_yb = emotion_yb.to(device)
        group_yb = group_yb.to(device)
        optimizer.zero_grad()
        emotion_logits, group_logits = model(stat_xb, seq_xb, adversarial_weight)
        emotion_loss = emotion_criterion(emotion_logits, emotion_yb)
        group_loss = group_criterion(group_logits, group_yb)
        loss = emotion_loss + group_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * emotion_yb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    emotion_true, emotion_pred = [], []
    group_true, group_pred = [], []
    for stat_xb, seq_xb, emotion_yb, group_yb in loader:
        stat_xb = stat_xb.to(device)
        seq_xb = seq_xb.to(device)
        emotion_logits, group_logits = model(stat_xb, seq_xb, 0.0)
        emotion_preds = torch.argmax(emotion_logits, dim=1).cpu().numpy()
        group_preds = torch.argmax(group_logits, dim=1).cpu().numpy()
        emotion_pred.extend(emotion_preds.tolist())
        emotion_true.extend(emotion_yb.numpy().tolist())
        group_pred.extend(group_preds.tolist())
        group_true.extend(group_yb.numpy().tolist())
    emotion_true_arr = np.array(emotion_true, dtype=np.int64)
    emotion_pred_arr = np.array(emotion_pred, dtype=np.int64)
    group_true_arr = np.array(group_true, dtype=np.int64)
    group_pred_arr = np.array(group_pred, dtype=np.int64)
    return {
        "emotion_accuracy": safe_accuracy(emotion_true_arr, emotion_pred_arr),
        "emotion_confusion_matrix": confusion_matrix(emotion_true_arr, emotion_pred_arr).tolist(),
        "emotion_preds": emotion_pred_arr,
        "emotion_true": emotion_true_arr,
        "group_accuracy": safe_accuracy(group_true_arr, group_pred_arr),
        "group_confusion_matrix": confusion_matrix(group_true_arr, group_pred_arr).tolist(),
        "group_preds": group_pred_arr,
        "group_true": group_true_arr,
    }


def maybe_standardize_inputs(stat_train, stat_valid, seq_train, seq_valid, args):
    stat_mean, stat_std = fit_stat_scaler(stat_train)
    stat_train = apply_stat_scaler(stat_train, stat_mean, stat_std)
    stat_valid = apply_stat_scaler(stat_valid, stat_mean, stat_std)
    stats = {"stat_mean": stat_mean, "stat_std": stat_std}
    if args.standardize_sequence_input:
        seq_mean, seq_std = fit_window_normalizer(seq_train)
        seq_train = apply_window_normalizer(seq_train, seq_mean, seq_std)
        seq_valid = apply_window_normalizer(seq_valid, seq_mean, seq_std)
        stats["seq_mean"] = seq_mean
        stats["seq_std"] = seq_std
    return stat_train, stat_valid, seq_train, seq_valid, stats


def train_one_split(stat_train, emotion_train, group_train, seq_train, stat_valid, emotion_valid, group_valid, seq_valid, args, device):
    stat_train, stat_valid, seq_train, seq_valid, stats = maybe_standardize_inputs(
        stat_train, stat_valid, seq_train, seq_valid, args
    )
    model = AdversarialDualBranchClassifier(
        stat_in_features=stat_train.shape[-1],
        seq_in_channels=seq_train.shape[-1],
        seq_num_nodes=seq_train.shape[2],
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        temporal_type=args.temporal_type,
        temporal_heads=args.temporal_heads,
        fusion_type=args.fusion_type,
        gate_mode=args.gate_mode,
        fusion_proj_dim=args.fusion_proj_dim,
        gru_layers=args.gru_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    emotion_criterion = nn.CrossEntropyLoss()
    group_criterion = nn.CrossEntropyLoss()
    train_loader = build_loader(stat_train, seq_train, emotion_train, group_train, args.batch_size, True)
    valid_loader = build_loader(stat_valid, seq_valid, emotion_valid, group_valid, args.batch_size, False)

    best_acc = -1.0
    best_state = None
    for _ in range(args.epochs):
        run_epoch(model, train_loader, optimizer, emotion_criterion, group_criterion, args.adversarial_weight, device)
        metrics = evaluate(model, valid_loader, device)
        if metrics["emotion_accuracy"] > best_acc:
            best_acc = metrics["emotion_accuracy"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, evaluate(model, valid_loader, device), stats


def assert_aligned(feature_meta, sequence_meta):
    if len(feature_meta) != len(sequence_meta):
        raise ValueError("Feature cache and sequence cache have different lengths.")
    for idx, (left, right) in enumerate(zip(feature_meta, sequence_meta, strict=True)):
        left_key = (str(left["subject_id"]), str(left["trial_id"]), int(left["label"]))
        right_key = (str(right["subject_id"]), str(right["trial_id"]), int(right["label"]))
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
    emotion_labels = np.array([int(item["label"]) for item in feature_meta], dtype=np.int64)
    group_labels = np.array([1 if str(item["group_name"]) == "抑郁症患者" else 0 for item in feature_meta], dtype=np.int64)
    groups = np.array([str(item["subject_id"]) for item in feature_meta], dtype=object)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splitter = GroupKFold(n_splits=args.folds)
    fold_results = []
    all_emotion_true, all_emotion_pred = [], []
    all_group_true, all_group_pred = [], []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, emotion_labels, groups), start=1):
        _, metrics, _ = train_one_split(
            features[train_idx], emotion_labels[train_idx], group_labels[train_idx], sequences[train_idx],
            features[valid_idx], emotion_labels[valid_idx], group_labels[valid_idx], sequences[valid_idx],
            args, device
        )
        fold_results.append(
            {
                "fold": fold_idx,
                "emotion_accuracy": metrics["emotion_accuracy"],
                "group_accuracy": metrics["group_accuracy"],
                "emotion_confusion_matrix": metrics["emotion_confusion_matrix"],
                "group_confusion_matrix": metrics["group_confusion_matrix"],
                "subjects": sorted(set(groups[valid_idx].tolist())),
                "emotion_group_accuracy": emotion_group_accuracy(feature_meta, valid_idx, metrics["emotion_preds"], emotion_labels[valid_idx]),
            }
        )
        all_emotion_true.extend(metrics["emotion_true"].tolist())
        all_emotion_pred.extend(metrics["emotion_preds"].tolist())
        all_group_true.extend(metrics["group_true"].tolist())
        all_group_pred.extend(metrics["group_preds"].tolist())
        print(f"Fold {fold_idx}: emotion_acc={metrics['emotion_accuracy']:.4f}, group_acc={metrics['group_accuracy']:.4f}")

    holdout_subjects = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    train_mask = np.array([group not in holdout_subjects for group in groups], dtype=bool)
    valid_mask = ~train_mask
    _, holdout_metrics, _ = train_one_split(
        features[train_mask], emotion_labels[train_mask], group_labels[train_mask], sequences[train_mask],
        features[valid_mask], emotion_labels[valid_mask], group_labels[valid_mask], sequences[valid_mask],
        args, device
    )
    holdout_emotion_group_accuracy = emotion_group_accuracy(
        feature_meta, np.where(valid_mask)[0], holdout_metrics["emotion_preds"], emotion_labels[valid_mask]
    )
    print(f"Holdout emotion_acc={holdout_metrics['emotion_accuracy']:.4f}, group_acc={holdout_metrics['group_accuracy']:.4f}")

    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    report = {
        "config": vars(args),
        "folds": fold_results,
        "cv_mean_emotion_accuracy": float(np.mean([item["emotion_accuracy"] for item in fold_results])),
        "cv_std_emotion_accuracy": float(np.std([item["emotion_accuracy"] for item in fold_results])),
        "cv_mean_group_accuracy": float(np.mean([item["group_accuracy"] for item in fold_results])),
        "cv_std_group_accuracy": float(np.std([item["group_accuracy"] for item in fold_results])),
        "overall_emotion_confusion_matrix": confusion_matrix(all_emotion_true, all_emotion_pred).tolist(),
        "overall_group_confusion_matrix": confusion_matrix(all_group_true, all_group_pred).tolist(),
        "holdout": {
            "subjects": sorted(holdout_subjects),
            "emotion_accuracy": holdout_metrics["emotion_accuracy"],
            "emotion_confusion_matrix": holdout_metrics["emotion_confusion_matrix"],
            "emotion_group_accuracy": holdout_emotion_group_accuracy,
            "emotion_group_gap_abs": abs(
                holdout_emotion_group_accuracy.get("抑郁症患者", 0.0) - holdout_emotion_group_accuracy.get("正常人", 0.0)
            ),
            "group_accuracy": holdout_metrics["group_accuracy"],
            "group_confusion_matrix": holdout_metrics["group_confusion_matrix"],
        },
    }
    report["config"]["root"] = str(args.root)
    report["config"]["feature_cache_path"] = str(feature_cache_path)
    report["config"]["sequence_cache_path"] = str(sequence_cache_path)
    report_path = paths.outputs_root / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
