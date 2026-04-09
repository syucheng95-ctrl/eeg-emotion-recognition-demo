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
from eeg_pipeline.graph_models import GraphSequenceClassifier
from eeg_pipeline.normalization import apply_window_normalizer, fit_window_normalizer
from train_graph_model import build_edge_index, group_accuracy
from train_gnn import HOLDOUT_DEFAULT, build_adjacency


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GraphDANN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_nodes: int, dropout: float):
        super().__init__()
        self.encoder = GraphSequenceClassifier(
            conv_type="gcn",
            temporal_type="meanstd",
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=2,
            num_nodes=num_nodes,
            dropout=dropout,
        )
        feature_dim = hidden_channels * 2
        self.label_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    def extract(self, x, base_adj, edge_index):
        sequence = self.encoder.encode_sequence(x, base_adj, edge_index)
        return self.encoder.pool_sequence(sequence)

    def forward(self, x, base_adj, edge_index, lambda_grl: float = 0.0):
        features = self.extract(x, base_adj, edge_index)
        class_logits = self.label_head(features)
        reversed_features = GradientReverse.apply(features, lambda_grl)
        domain_logits = self.domain_head(reversed_features)
        return class_logits, domain_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train graph DANN with public-test unlabeled target.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-cache", type=Path, default=None)
    parser.add_argument("--target-cache", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda-domain", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize-input", action="store_true")
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--report-name", type=str, default="graph_dann_report.json")
    parser.add_argument("--model-name", type=str, default="graph_dann.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_split(x_source_train, y_source_train, x_target_train, x_valid, y_valid, args, device, base_adj, edge_index):
    stats = None
    if args.standardize_input:
        mean, std = fit_window_normalizer(x_source_train)
        x_source_train = apply_window_normalizer(x_source_train, mean, std)
        x_target_train = apply_window_normalizer(x_target_train, mean, std)
        x_valid = apply_window_normalizer(x_valid, mean, std)
        stats = {"mean": mean, "std": std}
    model = GraphDANN(
        in_channels=x_source_train.shape[-1],
        hidden_channels=args.hidden_channels,
        num_nodes=x_source_train.shape[2],
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()
    source_loader = make_loader(x_source_train, y_source_train, args.batch_size, True)
    target_loader = make_loader(x_target_train, np.zeros(len(x_target_train), dtype=np.int64), args.batch_size, True)
    valid_loader = make_loader(x_valid, y_valid, args.batch_size, False)

    best_acc = -1.0
    best_state = None
    for _ in range(args.epochs):
        model.train()
        target_iter = iter(target_loader)
        for xs, ys in source_loader:
            try:
                xt, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt, _ = next(target_iter)
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            optimizer.zero_grad()
            class_logits, domain_logits_s = model(xs, base_adj, edge_index, lambda_grl=args.lambda_domain)
            _, domain_logits_t = model(xt, base_adj, edge_index, lambda_grl=args.lambda_domain)
            domain_labels_s = torch.zeros(xs.size(0), dtype=torch.long, device=device)
            domain_labels_t = torch.ones(xt.size(0), dtype=torch.long, device=device)
            loss = ce(class_logits, ys)
            loss = loss + ce(domain_logits_s, domain_labels_s) + ce(domain_logits_t, domain_labels_t)
            loss.backward()
            optimizer.step()

        acc, _, _ = evaluate(model, valid_loader, device, base_adj, edge_index)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    acc, cm, preds = evaluate(model, valid_loader, device, base_adj, edge_index)
    return model, acc, cm, preds, stats


@torch.no_grad()
def evaluate(model, loader, device, base_adj, edge_index):
    model.eval()
    all_true, all_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits, _ = model(xb, base_adj, edge_index, lambda_grl=0.0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.numpy().tolist())
    acc = float(accuracy_score(all_true, all_pred))
    cm = confusion_matrix(all_true, all_pred).tolist()
    return acc, cm, np.array(all_pred, dtype=np.int64)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = Paths(args.root.resolve())
    source_cache = args.source_cache or (paths.artifacts_root / "gnn_train_graph.npz")
    target_cache = args.target_cache or (paths.artifacts_root / "gnn_test_graph.npz")
    source_x, metadata = load_graph_cache(source_cache)
    target_x, _ = load_graph_cache(target_cache)
    labels = np.array([int(item["label"]) for item in metadata], dtype=np.int64)
    groups = np.array([str(item["subject_id"]) for item in metadata], dtype=object)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_adj = build_adjacency().to(device)
    edge_index = build_edge_index().to(device)

    splitter = GroupKFold(n_splits=args.folds)
    fold_results = []
    all_true, all_pred = [], []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(source_x, labels, groups), start=1):
        _, acc, cm, preds, _ = train_split(
            source_x[train_idx], labels[train_idx], target_x,
            source_x[valid_idx], labels[valid_idx], args, device, base_adj, edge_index
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
    _, holdout_acc, holdout_cm, holdout_preds, _ = train_split(
        source_x[train_mask], labels[train_mask], target_x,
        source_x[valid_mask], labels[valid_mask], args, device, base_adj, edge_index
    )
    print(f"Holdout acc={holdout_acc:.4f}")

    final_source_x = source_x
    final_target_x = target_x
    final_stats = None
    if args.standardize_input:
        mean, std = fit_window_normalizer(final_source_x)
        final_source_x = apply_window_normalizer(final_source_x, mean, std)
        final_target_x = apply_window_normalizer(final_target_x, mean, std)
        final_stats = {"mean": mean, "std": std}

    final_model = GraphDANN(
        in_channels=final_source_x.shape[-1],
        hidden_channels=args.hidden_channels,
        num_nodes=final_source_x.shape[2],
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()
    source_loader = make_loader(final_source_x, labels, args.batch_size, True)
    target_loader = make_loader(final_target_x, np.zeros(len(final_target_x), dtype=np.int64), args.batch_size, True)
    for _ in range(args.epochs):
        final_model.train()
        target_iter = iter(target_loader)
        for xs, ys in source_loader:
            try:
                xt, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xt, _ = next(target_iter)
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            optimizer.zero_grad()
            class_logits, domain_logits_s = final_model(xs, base_adj, edge_index, lambda_grl=args.lambda_domain)
            _, domain_logits_t = final_model(xt, base_adj, edge_index, lambda_grl=args.lambda_domain)
            domain_labels_s = torch.zeros(xs.size(0), dtype=torch.long, device=device)
            domain_labels_t = torch.ones(xt.size(0), dtype=torch.long, device=device)
            loss = ce(class_logits, ys)
            loss = loss + ce(domain_logits_s, domain_labels_s) + ce(domain_logits_t, domain_labels_t)
            loss.backward()
            optimizer.step()

    report = {
        "config": vars(args),
        "cv_mean_accuracy": float(np.mean([item["accuracy"] for item in fold_results])),
        "cv_std_accuracy": float(np.std([item["accuracy"] for item in fold_results])),
        "folds": fold_results,
        "overall_confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
        "holdout": {
            "subjects": sorted(holdout_subjects),
            "accuracy": holdout_acc,
            "confusion_matrix": holdout_cm,
            "group_accuracy": group_accuracy(metadata, np.where(valid_mask)[0], holdout_preds, labels[valid_mask]),
        },
    }
    report["config"] = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in report["config"].items()
    }
    report["config"]["source_cache"] = str(source_cache)
    report["config"]["target_cache"] = str(target_cache)
    report_path = paths.outputs_root / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    model_path = paths.outputs_root / args.model_name
    payload = {"state_dict": final_model.state_dict()}
    if final_stats is not None:
        payload["input_mean"] = final_stats["mean"]
        payload["input_std"] = final_stats["std"]
    torch.save(payload, model_path)
    print(f"Saved report to: {report_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
