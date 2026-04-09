from __future__ import annotations

import torch
from torch import nn

from .graph_models import TemporalSelfAttention


class WindowSequenceClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int,
        dropout: float,
        temporal_type: str = "meanstd",
        temporal_heads: int = 4,
    ):
        super().__init__()
        self.temporal_type = temporal_type
        flat_dim = num_nodes * in_channels
        self.window_encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if temporal_type == "mean":
            temporal_dim = hidden_channels
        elif temporal_type == "meanstd":
            temporal_dim = hidden_channels * 2
        elif temporal_type == "attention":
            self.temporal_attention = nn.Linear(hidden_channels, 1)
            temporal_dim = hidden_channels
        elif temporal_type == "gru":
            self.temporal_gru = nn.GRU(
                input_size=hidden_channels,
                hidden_size=hidden_channels,
                batch_first=True,
            )
            temporal_dim = hidden_channels
        elif temporal_type == "selfattn":
            self.temporal_encoder = TemporalSelfAttention(
                hidden_dim=hidden_channels,
                heads=temporal_heads,
                dropout=dropout,
            )
            self.temporal_attention = nn.Linear(hidden_channels, 1)
            temporal_dim = hidden_channels
        else:
            raise ValueError(f"Unsupported temporal_type: {temporal_type}")

        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_windows, num_nodes, in_channels = x.shape
        flat = x.reshape(batch_size, num_windows, num_nodes * in_channels)
        return self.window_encoder(flat)

    def pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_type == "mean":
            return x.mean(dim=1)
        if self.temporal_type == "meanstd":
            mean = x.mean(dim=1)
            std = x.std(dim=1, unbiased=False)
            return torch.cat([mean, std], dim=-1)
        if self.temporal_type == "attention":
            weights = torch.softmax(self.temporal_attention(x).squeeze(-1), dim=1)
            return torch.sum(x * weights.unsqueeze(-1), dim=1)
        if self.temporal_type == "gru":
            _, hidden = self.temporal_gru(x)
            return hidden[-1]
        if self.temporal_type == "selfattn":
            x = self.temporal_encoder(x)
            weights = torch.softmax(self.temporal_attention(x).squeeze(-1), dim=1)
            return torch.sum(x * weights.unsqueeze(-1), dim=1)
        raise RuntimeError("Unsupported temporal mode reached")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.encode_sequence(x)
        pooled = self.pool_sequence(sequence)
        return self.classifier(pooled)
