from __future__ import annotations

import torch
from torch import nn

from .graph_models import TemporalSelfAttention


class StatisticalBranchEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_channels: int,
        dropout: float,
        out_features: int | None = None,
        num_layers: int = 2,
    ):
        super().__init__()
        output_dim = out_features or hidden_channels
        if num_layers < 2:
            raise ValueError("StatisticalBranchEncoder requires at least 2 layers.")
        layers: list[nn.Module] = [
            nn.Linear(in_features, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(num_layers - 2, 0)):
            layers.extend(
                [
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.extend(
            [
                nn.Linear(hidden_channels, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SequenceBranchEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_nodes: int,
        dropout: float,
        temporal_type: str = "mean",
        temporal_heads: int = 4,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.temporal_type = temporal_type
        self.gru_layers = gru_layers
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
            self.output_dim = hidden_channels
        elif temporal_type == "meanstd":
            self.output_dim = hidden_channels * 2
        elif temporal_type == "attention":
            self.temporal_attention = nn.Linear(hidden_channels, 1)
            self.output_dim = hidden_channels
        elif temporal_type == "gru":
            self.temporal_gru = nn.GRU(
                input_size=hidden_channels,
                hidden_size=hidden_channels,
                batch_first=True,
                num_layers=gru_layers,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
            self.output_dim = hidden_channels
        elif temporal_type == "selfattn":
            self.temporal_encoder = TemporalSelfAttention(
                hidden_dim=hidden_channels,
                heads=temporal_heads,
                dropout=dropout,
            )
            self.temporal_attention = nn.Linear(hidden_channels, 1)
            self.output_dim = hidden_channels
        else:
            raise ValueError(f"Unsupported temporal_type: {temporal_type}")

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
        return self.pool_sequence(self.encode_sequence(x))


class DualBranchClassifier(nn.Module):
    def __init__(
        self,
        stat_in_features: int,
        seq_in_channels: int,
        seq_num_nodes: int,
        hidden_channels: int,
        dropout: float,
        temporal_type: str = "mean",
        temporal_heads: int = 4,
        fusion_type: str = "concat",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.stat_branch = StatisticalBranchEncoder(
            in_features=stat_in_features,
            hidden_channels=hidden_channels,
            dropout=dropout,
            out_features=hidden_channels,
        )
        self.seq_branch = SequenceBranchEncoder(
            in_channels=seq_in_channels,
            hidden_channels=hidden_channels,
            num_nodes=seq_num_nodes,
            dropout=dropout,
            temporal_type=temporal_type,
            temporal_heads=temporal_heads,
        )
        self.seq_projection = nn.Identity()
        seq_dim = self.seq_branch.output_dim
        if seq_dim != hidden_channels:
            self.seq_projection = nn.Sequential(
                nn.Linear(seq_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if fusion_type == "concat":
            classifier_input_dim = hidden_channels * 2
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.Sigmoid(),
            )
            classifier_input_dim = hidden_channels
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2),
        )

    def encode_branches(
        self,
        stat_x: torch.Tensor,
        seq_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stat_embedding = self.stat_branch(stat_x)
        seq_embedding = self.seq_projection(self.seq_branch(seq_x))
        return stat_embedding, seq_embedding

    def fuse_embeddings(
        self,
        stat_embedding: torch.Tensor,
        seq_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([stat_embedding, seq_embedding], dim=-1)
        gate = self.gate(torch.cat([stat_embedding, seq_embedding], dim=-1))
        return gate * stat_embedding + (1.0 - gate) * seq_embedding

    def forward(self, stat_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        stat_embedding, seq_embedding = self.encode_branches(stat_x, seq_x)
        fused = self.fuse_embeddings(stat_embedding, seq_embedding)
        return self.classifier(fused)
