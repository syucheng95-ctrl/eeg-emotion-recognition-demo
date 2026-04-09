from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import DenseGCNConv
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from eeg_pipeline.config import Paths
from eeg_pipeline.graph_features import load_graph_cache


HOLDOUT_DEFAULT = (
    "DEP1015,DEP1022,DEP1028,DEP1034,HC1010,HC1016,"
    "HC1024,HC1032,HC1038,HC1048,HC1054,HC1068"
)

CHANNELS = [
    "FP1","FP2","F7","F3","FZ","F4","F8","FT7","FC3","FCZ","FC4","FT8","T3","C3","CZ",
    "C4","T4","TP7","CP3","CPZ","CP4","TP8","T5","P3","PZ","P4","T6","O1","OZ","O2",
]


def build_edge_pairs() -> list[tuple[str, str]]:
    return [
        ("FP1", "FP2"), ("FP1", "F3"), ("FP1", "F7"), ("FP2", "F4"), ("FP2", "F8"),
        ("F7", "F3"), ("F3", "FZ"), ("FZ", "F4"), ("F4", "F8"),
        ("F7", "FT7"), ("F3", "FC3"), ("FZ", "FCZ"), ("F4", "FC4"), ("F8", "FT8"),
        ("FT7", "FC3"), ("FC3", "FCZ"), ("FCZ", "FC4"), ("FC4", "FT8"),
        ("FT7", "T3"), ("FC3", "C3"), ("FCZ", "CZ"), ("FC4", "C4"), ("FT8", "T4"),
        ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"),
        ("T3", "TP7"), ("C3", "CP3"), ("CZ", "CPZ"), ("C4", "CP4"), ("T4", "TP8"),
        ("TP7", "CP3"), ("CP3", "CPZ"), ("CPZ", "CP4"), ("CP4", "TP8"),
        ("TP7", "T5"), ("CP3", "P3"), ("CPZ", "PZ"), ("CP4", "P4"), ("TP8", "T6"),
        ("T5", "P3"), ("P3", "PZ"), ("PZ", "P4"), ("P4", "T6"),
        ("T5", "O1"), ("P3", "O1"), ("PZ", "OZ"), ("P4", "O2"), ("T6", "O2"),
        ("O1", "OZ"), ("OZ", "O2"),
    ]


def build_adjacency() -> torch.Tensor:
    index = {name: idx for idx, name in enumerate(CHANNELS)}
    adj = torch.eye(len(CHANNELS), dtype=torch.float32)
    for left, right in build_edge_pairs():
        i, j = index[left], index[right]
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


class TrialGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.3,
        num_layers: int = 2,
        window_pool: str = "mean",
    ):
        super().__init__()
        self.input_conv = DenseGCNConv(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        self.hidden_convs = nn.ModuleList(
            [DenseGCNConv(hidden_channels, hidden_channels) for _ in range(max(num_layers - 1, 0))]
        )
        self.hidden_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(max(num_layers - 1, 0))]
        )
        self.dropout = nn.Dropout(dropout)
        self.window_pool = window_pool
        if window_pool == "attention":
            self.window_attention = nn.Linear(hidden_channels, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, W, N, F]
        bsz, num_windows, num_nodes, in_channels = x.shape
        x = x.reshape(bsz * num_windows, num_nodes, in_channels)
        x = self.input_conv(x, adj)
        x = self.input_norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        for conv, norm in zip(self.hidden_convs, self.hidden_norms, strict=True):
            residual = x
            x = conv(x, adj)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + residual
        x = x.mean(dim=1)  # node pooling
        x = x.reshape(bsz, num_windows, -1)
        if self.window_pool == "mean":
            x = x.mean(dim=1)
        elif self.window_pool == "attention":
            weights = torch.softmax(self.window_attention(x).squeeze(-1), dim=1)
            x = torch.sum(x * weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unsupported window_pool: {self.window_pool}")
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN baseline with subject-level validation.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--window-pool", type=str, default="mean", choices=["mean", "attention"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--report-name", type=str, default="gnn_cv_report.json")
    parser.add_argument("--model-name", type=str, default="gnn_model.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, criterion, device, adj):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb, adj)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, adj):
    model.eval()
    all_true, all_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb, adj)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.numpy().tolist())
    acc = float(accuracy_score(all_true, all_pred))
    cm = confusion_matrix(all_true, all_pred).tolist()
    return acc, cm, np.array(all_pred, dtype=np.int64)


def train_one_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    adj: torch.Tensor,
):
    model = TrialGCN(
        in_channels=x_train.shape[-1],
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        num_layers=args.num_layers,
        window_pool=args.window_pool,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = build_loader(x_train, y_train, args.batch_size, shuffle=True)
    valid_loader = build_loader(x_valid, y_valid, args.batch_size, shuffle=False)

    best_acc = -1.0
    best_state = None
    for _ in range(args.epochs):
        run_epoch(model, train_loader, optimizer, criterion, device, adj)
        acc, _, _ = evaluate(model, valid_loader, device, adj)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    acc, cm, preds = evaluate(model, valid_loader, device, adj)
    return model, acc, cm, preds


def group_accuracy(metadata, indices, preds, labels):
    result = {}
    group_names = np.array([metadata[idx]["group_name"] for idx in indices], dtype=object)
    for group_name in sorted(set(group_names.tolist())):
        mask = group_names == group_name
        result[group_name] = float(accuracy_score(labels[mask], preds[mask]))
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    paths = Paths(args.root.resolve())
    cache_path = args.cache_path or (paths.artifacts_root / "gnn_train_graph.npz")
    tensors, metadata = load_graph_cache(cache_path)
    labels = np.array([int(item["label"]) for item in metadata], dtype=np.int64)
    groups = np.array([str(item["subject_id"]) for item in metadata], dtype=object)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = build_adjacency().to(device)

    splitter = GroupKFold(n_splits=args.folds)
    fold_results = []
    all_true, all_pred = [], []
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(tensors, labels, groups), start=1):
        _, acc, cm, preds = train_one_split(
            tensors[train_idx],
            labels[train_idx],
            tensors[valid_idx],
            labels[valid_idx],
            args,
            device,
            adj,
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
    holdout_model, holdout_acc, holdout_cm, holdout_preds = train_one_split(
        tensors[train_mask],
        labels[train_mask],
        tensors[valid_mask],
        labels[valid_mask],
        args,
        device,
        adj,
    )
    print(f"Holdout acc={holdout_acc:.4f}")

    final_model, _, _, _ = train_one_split(
        tensors,
        labels,
        tensors,
        labels,
        args,
        device,
        adj,
    )

    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    report = {
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
        "config": {
            "cache_path": str(cache_path),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "device": str(device),
        },
    }
    report_path = paths.outputs_root / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    model_path = paths.outputs_root / args.model_name
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "adjacency": build_adjacency(),
            "in_channels": tensors.shape[-1],
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
            "window_pool": args.window_pool,
        },
        model_path,
    )
    print(f"Saved report to: {report_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
