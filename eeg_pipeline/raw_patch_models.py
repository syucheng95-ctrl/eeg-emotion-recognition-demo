from __future__ import annotations

import torch
from torch import nn

from .graph_models import GraphBackbone


class RawPatchChannelEncoder(nn.Module):
    def __init__(self, patch_samples: int, channel_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(patch_samples, channel_hidden),
            nn.LayerNorm(channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RawPatchClassifier(nn.Module):
    def __init__(
        self,
        variant: str,
        patch_samples: int,
        num_nodes: int,
        channel_hidden: int,
        graph_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.variant = variant
        self.channel_encoder = RawPatchChannelEncoder(
            patch_samples=patch_samples,
            channel_hidden=channel_hidden,
            dropout=dropout,
        )
        if variant in {"b", "c"}:
            self.graph_backbone = GraphBackbone(
                conv_type="gcn",
                in_channels=channel_hidden,
                hidden_channels=graph_hidden,
                num_layers=2,
                num_nodes=num_nodes,
                dropout=dropout,
            )
            patch_dim = graph_hidden
        else:
            patch_dim = channel_hidden

        if variant == "c":
            self.temporal_model = nn.GRU(
                input_size=patch_dim,
                hidden_size=graph_hidden,
                batch_first=True,
            )
            final_dim = graph_hidden
        else:
            final_dim = patch_dim

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, graph_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden, 2),
        )

    def encode_patches(self, x: torch.Tensor, base_adj: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        encoded = self.channel_encoder(x)
        if self.variant in {"b", "c"}:
            encoded = self.graph_backbone(encoded, base_adj, edge_index)
        return encoded.mean(dim=2)

    def forward(self, x: torch.Tensor, base_adj: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        patch_sequence = self.encode_patches(x, base_adj, edge_index)
        if self.variant == "c":
            _, hidden = self.temporal_model(patch_sequence)
            pooled = hidden[-1]
        else:
            pooled = patch_sequence.mean(dim=1)
        return self.classifier(pooled)
