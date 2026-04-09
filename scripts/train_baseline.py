from __future__ import annotations

import _bootstrap

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eeg_pipeline.config import Paths
from eeg_pipeline.features import load_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subject-level cross validation baselines.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root path.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Optional feature cache path. Defaults to artifacts/train_features.npz",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--holdout-subjects",
        type=str,
        default="",
        help="Comma-separated subject ids reserved as pseudo-private validation set.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        choices=["logreg", "svm", "rf"],
        help="Model to refit on all training data after cross-validation.",
    )
    return parser.parse_args()


def build_models() -> dict[str, object]:
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True, random_state=42)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
    }


def _safe_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float(accuracy_score(y_true, y_pred))
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = float(np.mean(y_pred[pos_mask] == 1)) if np.any(pos_mask) else 0.0
    tnr = float(np.mean(y_pred[neg_mask] == 0)) if np.any(neg_mask) else 0.0
    balanced_acc = 0.5 * (tpr + tnr)
    return {"accuracy": accuracy, "balanced_accuracy": balanced_acc}


def _group_metrics(
    metadata: list[dict[str, object]],
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    names = [str(metadata[idx]["group_name"]) for idx in indices]
    results: dict[str, dict[str, float]] = {}
    for group_name in sorted(set(names)):
        mask = np.array([name == group_name for name in names], dtype=bool)
        results[group_name] = _safe_binary_metrics(y_true[mask], y_pred[mask])
    return results


def run_cv(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    metadata: list[dict[str, object]],
) -> dict[str, object]:
    splitter = GroupKFold(n_splits=folds)
    models = build_models()
    results: dict[str, object] = {"folds": folds, "models": {}}
    for model_name, model in models.items():
        fold_metrics: list[dict[str, object]] = []
        all_true: list[int] = []
        all_pred: list[int] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, labels, groups), start=1):
            model.fit(features[train_idx], labels[train_idx])
            preds = model.predict(features[valid_idx])
            group_metrics = _group_metrics(metadata, valid_idx, labels[valid_idx], preds)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    **_safe_binary_metrics(labels[valid_idx], preds),
                    "subjects": sorted(set(groups[valid_idx].tolist())),
                    "group_metrics": group_metrics,
                }
            )
            all_true.extend(labels[valid_idx].tolist())
            all_pred.extend(preds.tolist())
        report = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
        results["models"][model_name] = {
            "fold_metrics": fold_metrics,
            "mean_accuracy": float(np.mean([item["accuracy"] for item in fold_metrics])),
            "std_accuracy": float(np.std([item["accuracy"] for item in fold_metrics])),
            "mean_balanced_accuracy": float(
                np.mean([item["balanced_accuracy"] for item in fold_metrics])
            ),
            "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
            "classification_report": report,
        }
    return results


def refit_and_save_best(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    model_output_path: Path,
) -> None:
    model = build_models()[model_name]
    model.fit(features, labels)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with model_output_path.open("wb") as handle:
        pickle.dump(model, handle)


def run_holdout(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    metadata: list[dict[str, object]],
    holdout_subjects: set[str],
) -> dict[str, object]:
    train_mask = np.array([group not in holdout_subjects for group in groups], dtype=bool)
    valid_mask = ~train_mask
    if not np.any(valid_mask):
        raise ValueError("No holdout subjects matched the provided ids.")
    if not np.any(train_mask):
        raise ValueError("Holdout split consumed all subjects.")

    models = build_models()
    results: dict[str, object] = {"holdout_subjects": sorted(holdout_subjects), "models": {}}
    for model_name, model in models.items():
        model.fit(features[train_mask], labels[train_mask])
        preds = model.predict(features[valid_mask])
        report = classification_report(labels[valid_mask], preds, output_dict=True, zero_division=0)
        results["models"][model_name] = {
            **_safe_binary_metrics(labels[valid_mask], preds),
            "confusion_matrix": confusion_matrix(labels[valid_mask], preds).tolist(),
            "classification_report": report,
            "group_metrics": _group_metrics(
                metadata,
                np.where(valid_mask)[0],
                labels[valid_mask],
                preds,
            ),
        }
    return results


def main() -> None:
    args = parse_args()
    paths = Paths(args.root.resolve())
    cache_path = args.cache_path or (paths.artifacts_root / "train_features.npz")
    features, metadata = load_feature_cache(cache_path)
    labels = np.array([int(item["label"]) for item in metadata], dtype=np.int64)
    groups = np.array([str(item["subject_id"]) for item in metadata], dtype=object)

    if len(np.unique(groups)) < args.folds:
        raise ValueError(f"Need at least {args.folds} unique subjects for GroupKFold.")

    results = run_cv(features, labels, groups, args.folds, metadata)
    holdout_results = None
    if args.holdout_subjects.strip():
        holdout_subjects = {
            item.strip() for item in args.holdout_subjects.split(",") if item.strip()
        }
        holdout_results = run_holdout(features, labels, groups, metadata, holdout_subjects)

    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    report_path = paths.outputs_root / "baseline_cv_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        payload = {"cross_validation": results, "holdout": holdout_results}
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    best_model_name = max(
        results["models"].items(),
        key=lambda item: item[1]["mean_accuracy"],
    )[0]
    chosen_model = args.model or best_model_name
    refit_and_save_best(chosen_model, features, labels, paths.outputs_root / "baseline_model.pkl")

    print(f"Cross-validation report saved to: {report_path}")
    for model_name, metric in results["models"].items():
        print(
            f"{model_name}: mean_acc={metric['mean_accuracy']:.4f}, "
            f"balanced_acc={metric['mean_balanced_accuracy']:.4f}, "
            f"std={metric['std_accuracy']:.4f}"
        )
    if holdout_results is not None:
        print("Holdout evaluation:")
        for model_name, metric in holdout_results["models"].items():
            print(
                f"{model_name}: holdout_acc={metric['accuracy']:.4f}, "
                f"holdout_bal_acc={metric['balanced_accuracy']:.4f}"
            )
    print(f"Refit model saved to: {paths.outputs_root / 'baseline_model.pkl'}")


if __name__ == "__main__":
    main()
