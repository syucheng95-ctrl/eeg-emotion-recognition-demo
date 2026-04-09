from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_pipeline.config import Paths
from eeg_pipeline.features import load_feature_cache
from eeg_pipeline.graph_features import load_graph_cache
from eeg_pipeline.multitask_models import MultiTaskDualBranchClassifier
from eeg_pipeline.normalization import apply_window_normalizer, fit_window_normalizer
from train_gnn import HOLDOUT_DEFAULT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate seed-ensemble strategies for the finalized phase-14 multitask dual-branch model."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--feature-cache-path", type=Path, default=None)
    parser.add_argument("--sequence-cache-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--holdout-subjects", type=str, default=HOLDOUT_DEFAULT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--summary-name", type=str, default="final_strategy_summary.csv")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(
    stat_x: np.ndarray,
    seq_x: np.ndarray,
    emotion_y: np.ndarray,
    group_y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
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


def assert_aligned(feature_meta, sequence_meta) -> None:
    if len(feature_meta) != len(sequence_meta):
        raise ValueError("Feature cache and sequence cache have different lengths.")
    for idx, (left, right) in enumerate(zip(feature_meta, sequence_meta, strict=True)):
        left_key = (str(left["subject_id"]), str(left["trial_id"]), int(left["label"]))
        right_key = (str(right["subject_id"]), str(right["trial_id"]), int(right["label"]))
        if left_key != right_key:
            raise ValueError(f"Cache mismatch at index {idx}: {left_key} != {right_key}")


def safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(accuracy_score(y_true, y_pred))


def group_accuracy(metadata, indices, y_true, y_pred):
    result = {}
    group_names = np.array([metadata[idx]["group_name"] for idx in indices], dtype=object)
    for group_name in sorted(set(group_names.tolist())):
        mask = group_names == group_name
        result[group_name] = safe_accuracy(y_true[mask], y_pred[mask])
    return result


def run_epoch(model, loader, optimizer, emotion_criterion, group_criterion, group_loss_weight, device):
    model.train()
    for stat_xb, seq_xb, emotion_yb, group_yb in loader:
        stat_xb = stat_xb.to(device)
        seq_xb = seq_xb.to(device)
        emotion_yb = emotion_yb.to(device)
        group_yb = group_yb.to(device)
        optimizer.zero_grad()
        emotion_logits, group_logits = model(stat_xb, seq_xb)
        emotion_loss = emotion_criterion(emotion_logits, emotion_yb)
        group_loss = group_criterion(group_logits, group_yb)
        loss = emotion_loss + group_loss_weight * group_loss
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict_holdout_proba(model, stat_x, seq_x, batch_size, device) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(stat_x).float(), torch.from_numpy(seq_x).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    probs = []
    for stat_xb, seq_xb in loader:
        stat_xb = stat_xb.to(device)
        seq_xb = seq_xb.to(device)
        emotion_logits, _ = model(stat_xb, seq_xb)
        probs.append(torch.softmax(emotion_logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_candidate_and_predict(
    features: np.ndarray,
    sequences: np.ndarray,
    emotion_labels: np.ndarray,
    group_labels: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    args: argparse.Namespace,
    exp: dict[str, object],
    device: torch.device,
) -> np.ndarray:
    train_features = features[train_mask]
    valid_features = features[valid_mask]
    train_sequences = sequences[train_mask]
    valid_sequences = sequences[valid_mask]

    stat_mean, stat_std = fit_stat_scaler(train_features)
    train_features = apply_stat_scaler(train_features, stat_mean, stat_std)
    valid_features = apply_stat_scaler(valid_features, stat_mean, stat_std)

    if bool(exp["standardize_sequence_input"]):
        seq_mean, seq_std = fit_window_normalizer(train_sequences)
        train_sequences = apply_window_normalizer(train_sequences, seq_mean, seq_std)
        valid_sequences = apply_window_normalizer(valid_sequences, seq_mean, seq_std)

    model = MultiTaskDualBranchClassifier(
        stat_in_features=train_features.shape[-1],
        seq_in_channels=train_sequences.shape[-1],
        seq_num_nodes=train_sequences.shape[2],
        hidden_channels=int(exp["hidden_channels"]),
        dropout=float(exp["dropout"]),
        temporal_type=str(exp["temporal_type"]),
        temporal_heads=int(exp["temporal_heads"]),
        fusion_type=str(exp["fusion_type"]),
        gate_mode=str(exp["gate_mode"]),
        fusion_proj_dim=int(exp["fusion_proj_dim"]),
        gru_layers=int(exp["gru_layers"]),
        stat_hidden_channels=None if int(exp["stat_hidden_channels"]) <= 0 else int(exp["stat_hidden_channels"]),
        stat_num_layers=int(exp["stat_num_layers"]),
        interaction_mode=str(exp["interaction_mode"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    emotion_criterion = nn.CrossEntropyLoss()
    group_criterion = nn.CrossEntropyLoss()
    train_loader = build_loader(
        train_features,
        train_sequences,
        emotion_labels[train_mask],
        group_labels[train_mask],
        args.batch_size,
        shuffle=True,
    )
    for _ in range(int(exp["epochs"])):
        run_epoch(
            model,
            train_loader,
            optimizer,
            emotion_criterion,
            group_criterion,
            float(exp["group_loss_weight"]),
            device,
        )
    return predict_holdout_proba(model, valid_features, valid_sequences, args.batch_size, device)


def evaluate_probs(metadata, indices, labels, probs):
    preds = (probs >= 0.5).astype(np.int64)
    metrics = group_accuracy(metadata, indices, labels, preds)
    return {
        "holdout_accuracy": float(accuracy_score(labels, preds)),
        "holdout_dep_accuracy": metrics.get("抑郁症患者"),
        "holdout_hc_accuracy": metrics.get("正常人"),
        "holdout_group_gap_abs": abs(
            float(metrics.get("抑郁症患者", 0.0)) - float(metrics.get("正常人", 0.0))
        ),
    }


def weighted_average(prob_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr = weights_arr / np.sum(weights_arr)
    stacked = np.stack(prob_list, axis=0)
    return np.tensordot(weights_arr, stacked, axes=(0, 0))


def majority_vote(prob_list: list[np.ndarray]) -> np.ndarray:
    preds = np.stack([(probs >= 0.5).astype(np.int64) for probs in prob_list], axis=0)
    votes = preds.mean(axis=0)
    return votes.astype(np.float32)


def build_model_specs(outputs_root: Path) -> list[dict[str, object]]:
    specs = [
        {
            "name": "final_seqgate_seed42",
            "report_file": "interaction_seq_gate_report.json",
            "family": "final_seqgate",
            "seed": 42,
        },
        {
            "name": "final_seqgate_seed52",
            "report_file": "phase14_seqgate_lambda0p02_seed52_report.json",
            "family": "final_seqgate",
            "seed": 52,
        },
        {
            "name": "final_seqgate_seed62",
            "report_file": "phase14_seqgate_lambda0p02_seed62_report.json",
            "family": "final_seqgate",
            "seed": 62,
        },
    ]
    for spec in specs:
        report = json.loads((outputs_root / str(spec["report_file"])).read_text(encoding="utf-8"))
        cfg = report["config"]
        spec.update(
            {
                "hidden_channels": int(cfg["hidden_channels"]),
                "dropout": float(cfg["dropout"]),
                "temporal_type": str(cfg["temporal_type"]),
                "temporal_heads": int(cfg["temporal_heads"]),
                "fusion_type": str(cfg["fusion_type"]),
                "gate_mode": str(cfg.get("gate_mode", "vector")),
                "fusion_proj_dim": int(cfg.get("fusion_proj_dim", 0)),
                "gru_layers": int(cfg.get("gru_layers", 1)),
                "stat_hidden_channels": int(cfg.get("stat_hidden_channels", 0)),
                "stat_num_layers": int(cfg.get("stat_num_layers", 2)),
                "interaction_mode": str(cfg.get("interaction_mode", "none")),
                "group_loss_weight": float(cfg["group_loss_weight"]),
                "standardize_sequence_input": bool(cfg["standardize_sequence_input"]),
                "epochs": int(cfg["epochs"]),
                "cv_mean_emotion_accuracy": float(report["cv_mean_emotion_accuracy"]),
            }
        )
    return specs


def main() -> None:
    args = parse_args()
    paths = Paths(args.root.resolve())
    feature_cache_path = args.feature_cache_path or (paths.artifacts_root / "w2_s2_meanstd_classic5_train_features.npz")
    sequence_cache_path = args.sequence_cache_path or (paths.artifacts_root / "gnn_train_graph.npz")
    features, feature_meta = load_feature_cache(feature_cache_path)
    sequences, sequence_meta = load_graph_cache(sequence_cache_path)
    assert_aligned(feature_meta, sequence_meta)
    emotion_labels = np.array([int(item["label"]) for item in feature_meta], dtype=np.int64)
    group_labels = np.array([1 if str(item["group_name"]) == "抑郁症患者" else 0 for item in feature_meta], dtype=np.int64)
    groups = np.array([str(item["subject_id"]) for item in feature_meta], dtype=object)
    holdout_subjects = {item.strip() for item in args.holdout_subjects.split(",") if item.strip()}
    train_mask = np.array([group not in holdout_subjects for group in groups], dtype=bool)
    valid_mask = ~train_mask
    valid_indices = np.where(valid_mask)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_specs = build_model_specs(paths.outputs_root)
    prob_map: dict[str, np.ndarray] = {}
    for spec in model_specs:
        print(f"Training holdout-only prediction for {spec['name']}")
        set_seed(int(spec["seed"]))
        prob_map[str(spec["name"])] = train_candidate_and_predict(
            features,
            sequences,
            emotion_labels,
            group_labels,
            train_mask,
            valid_mask,
            args,
            spec,
            device,
        )

    all_names = [str(spec["name"]) for spec in model_specs]
    spec_map = {str(spec["name"]): spec for spec in model_specs}
    holdout_labels = emotion_labels[valid_mask]
    experiments = [
        {"ensemble_name": "final_seqgate_seed_avg", "members": all_names, "method": "soft_avg_equal"},
        {"ensemble_name": "final_seqgate_cv_weighted", "members": all_names, "method": "soft_avg_cv_weighted"},
        {"ensemble_name": "final_seqgate_majority_vote", "members": all_names, "method": "hard_vote"},
    ]

    results = []
    for exp in experiments:
        members = [str(name) for name in exp["members"]]
        prob_list = [prob_map[name] for name in members]
        if exp["method"] == "soft_avg_equal":
            fused = np.mean(np.stack(prob_list, axis=0), axis=0)
        elif exp["method"] == "soft_avg_cv_weighted":
            weights = [float(spec_map[name]["cv_mean_emotion_accuracy"]) for name in members]
            fused = weighted_average(prob_list, weights)
        elif exp["method"] == "hard_vote":
            fused = majority_vote(prob_list)
        else:
            raise ValueError(f"Unsupported method: {exp['method']}")
        metrics = evaluate_probs(feature_meta, valid_indices, holdout_labels, fused)
        results.append(
            {
                "ensemble_name": exp["ensemble_name"],
                "method": exp["method"],
                "num_members": len(members),
                "members": ",".join(members),
                **metrics,
            }
        )

    summary = pd.DataFrame(results).sort_values(
        ["holdout_accuracy", "holdout_group_gap_abs"],
        ascending=[False, True],
    )
    summary_path = paths.outputs_root / args.summary_name
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
