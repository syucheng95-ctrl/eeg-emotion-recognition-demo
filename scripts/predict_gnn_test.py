from __future__ import annotations

import _bootstrap

import argparse
from pathlib import Path

import pandas as pd
import torch

from eeg_pipeline.config import Paths
from eeg_pipeline.graph_features import load_graph_cache
from train_gnn import TrialGCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict public test set with trained GNN model.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(args.root.resolve())
    model_path = args.model_path or (paths.outputs_root / "gnn_model.pt")
    cache_path = args.cache_path or (paths.artifacts_root / "gnn_test_graph.npz")

    checkpoint = torch.load(model_path, map_location="cpu")
    model = TrialGCN(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
        dropout=checkpoint["dropout"],
        num_layers=checkpoint.get("num_layers", 2),
        window_pool=checkpoint.get("window_pool", "mean"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    adj = checkpoint["adjacency"]

    tensors, metadata = load_graph_cache(cache_path)
    x = torch.from_numpy(tensors).float()
    with torch.no_grad():
        logits = model(x, adj)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    rows = []
    for meta, pred in zip(metadata, preds, strict=True):
        rows.append(
            {
                "user_id": meta["subject_id"],
                "trial_id": int(meta["trial_id"]),
                "Emotion_label": int(pred),
            }
        )
    submission = pd.DataFrame(rows).sort_values(["user_id", "trial_id"]).reset_index(drop=True)
    out_path = paths.outputs_root / "submission_public_test_gnn.xlsx"
    submission.to_excel(out_path, index=False)
    print(f"Saved submission to: {out_path}")


if __name__ == "__main__":
    main()
