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
from eeg_pipeline.graph_features import load_graph_cache
from eeg_pipeline.normalization import apply_window_normalizer, fit_window_normalizer
from eeg_pipeline.window_models import WindowSequenceClassifier
from train_gnn import HOLDOUT_DEFAULT
from train_graph_model import group_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train non-graph window model with subject-level validation.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temporal-type", type=str, default="meanstd", choices=["mean", "attention", "meanstd", "gru", "selfattn"])
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize-input", action="store_true")
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--report-name", type=str, default="window_model_report.json")
    parser.add_argument("--model-name", type=str, default="window_model.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.numpy().tolist())
    acc = float(accuracy_score(all_true, all_pred))
    cm = confusion_matrix(all_true, all_pred).tolist()
    return acc, cm, np.array(all_pred, dtype=np.int64)


def maybe_standardize(train_x, valid_x, args):
    stats = None
    if args.standardize_input:
        mean, std = fit_window_normalizer(train_x)
        train_x = apply_window_normalizer(train_x, mean, std)
        valid_x = apply_window_normalizer(valid_x, mean, std)
        stats = {"mean": mean, "std": std}
    return train_x, valid_x, stats


def train_one_split(x_train, y_train, x_valid, y_valid, args, device):
    x_train, x_valid, stats = maybe_standardize(x_train, x_valid, args)
    model = WindowSequenceClassifier(
        in_channels=x_train.shape[-1],
        hidden_channels=args.hidden_channels,
        num_nodes=x_train.shape[2],
        dropout=args.dropout,
        temporal_type=args.temporal_type,
        temporal_heads=args.temporal_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = build_loader(x_train, y_train, args.batch_size, shuffle=True)
    valid_loader = build_loader(x_valid, y_valid, args.batch_size, shuffle=False)

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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = Paths(args.root.resolve())
    cache_path = args.cache_path or (paths.artifacts_root / "gnn_train_graph.npz")
    tensors, metadata = load_graph_cache(cache_path)
    labels = np.array([int(item["label"]) for item in metadata], dtype=np.int64)
    groups = np.array([str(item["subject_id"]) for item in metadata], dtype=object)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splitter = GroupKFold(n_splits=args.folds)
    fold_results = []
    all_true, all_pred = [], []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(tensors, labels, groups), start=1):
        _, acc, cm, preds, _ = train_one_split(
            tensors[train_idx],
            labels[train_idx],
            tensors[valid_idx],
            labels[valid_idx],
            args,
            device,
        )
        fold_results.append(
            {
                "fold": fold_idx,
                "accuracy": acc,
                "subjects": sorted(set(groups[valid_idx].tolist())),
                "confusion_matrix": cm,
                "group_accuracy": group_accuracy(metadata, valid_idx, preds, labels[valid_idx]),
            }
        )
        all_true.extend(labels[valid_idx].tolist())
        all_pred.extend(preds.tolist())
        print(f"Fold {fold_idx}: acc={acc:.4f}")

    holdout_subjects = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    train_mask = np.array([group not in holdout_subjects for group in groups], dtype=bool)
    valid_mask = ~train_mask
    _, holdout_acc, holdout_cm, holdout_preds, _ = train_one_split(
        tensors[train_mask],
        labels[train_mask],
        tensors[valid_mask],
        labels[valid_mask],
        args,
        device,
    )
    print(f"Holdout acc={holdout_acc:.4f}")

    final_tensors = tensors
    final_stats = None
    if args.standardize_input:
        mean, std = fit_window_normalizer(final_tensors)
        final_tensors = apply_window_normalizer(final_tensors, mean, std)
        final_stats = {"mean": mean, "std": std}
    final_model = WindowSequenceClassifier(
        in_channels=final_tensors.shape[-1],
        hidden_channels=args.hidden_channels,
        num_nodes=final_tensors.shape[2],
        dropout=args.dropout,
        temporal_type=args.temporal_type,
        temporal_heads=args.temporal_heads,
    ).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    final_loader = build_loader(final_tensors, labels, args.batch_size, shuffle=True)
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
            "group_accuracy": group_accuracy(metadata, np.where(valid_mask)[0], holdout_preds, labels[valid_mask]),
        },
    }
    report["config"]["root"] = str(args.root)
    report["config"]["cache_path"] = str(cache_path)
    report_path = paths.outputs_root / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    model_path = paths.outputs_root / args.model_name
    payload = {
        "state_dict": final_model.state_dict(),
        "in_channels": final_tensors.shape[-1],
        "hidden_channels": args.hidden_channels,
        "num_nodes": final_tensors.shape[2],
        "dropout": args.dropout,
        "temporal_type": args.temporal_type,
        "temporal_heads": args.temporal_heads,
    }
    if final_stats is not None:
        payload["input_mean"] = final_stats["mean"]
        payload["input_std"] = final_stats["std"]
    torch.save(payload, model_path)
    print(f"Saved report to: {report_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
