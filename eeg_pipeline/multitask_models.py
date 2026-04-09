from __future__ import annotations

import torch
from torch import nn

from .dual_branch_models import SequenceBranchEncoder, StatisticalBranchEncoder


class MultiTaskDualBranchClassifier(nn.Module):
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
        gate_mode: str = "vector",
        fusion_proj_dim: int = 0,
        gru_layers: int = 1,
        stat_hidden_channels: int | None = None,
        stat_num_layers: int = 2,
        interaction_mode: str = "none",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.gate_mode = gate_mode
        self.interaction_mode = interaction_mode
        stat_hidden = stat_hidden_channels or hidden_channels
        self.stat_branch = StatisticalBranchEncoder(
            in_features=stat_in_features,
            hidden_channels=stat_hidden,
            dropout=dropout,
            out_features=hidden_channels,
            num_layers=stat_num_layers,
        )
        self.seq_branch = SequenceBranchEncoder(
            in_channels=seq_in_channels,
            hidden_channels=hidden_channels,
            num_nodes=seq_num_nodes,
            dropout=dropout,
            temporal_type=temporal_type,
            temporal_heads=temporal_heads,
            gru_layers=gru_layers,
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
        if interaction_mode == "none":
            self.interaction = None
        elif interaction_mode == "seq_gate":
            self.interaction = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Sigmoid(),
            )
        elif interaction_mode == "film":
            self.interaction = nn.Linear(hidden_channels, hidden_channels * 2)
        elif interaction_mode == "seq_rescale":
            self.interaction = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported interaction_mode: {interaction_mode}")

        if fusion_type == "concat":
            shared_dim = hidden_channels * 2
        elif fusion_type == "gated":
            fusion_dim = fusion_proj_dim or hidden_channels
            self.stat_fusion_projection = nn.Identity()
            self.seq_fusion_projection = nn.Identity()
            if fusion_dim != hidden_channels:
                self.stat_fusion_projection = nn.Sequential(
                    nn.Linear(hidden_channels, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.seq_fusion_projection = nn.Sequential(
                    nn.Linear(hidden_channels, fusion_dim),
                    nn.LayerNorm(fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            self.fusion_dim = fusion_dim
            if gate_mode == "vector":
                self.gate = nn.Sequential(
                    nn.Linear(fusion_dim * 2, fusion_dim),
                    nn.Sigmoid(),
                )
            elif gate_mode == "scalar":
                self.gate = nn.Sequential(
                    nn.Linear(fusion_dim * 2, 1),
                    nn.Sigmoid(),
                )
            elif gate_mode == "dual":
                self.gate = nn.Sequential(
                    nn.Linear(fusion_dim * 2, fusion_dim * 2),
                    nn.Sigmoid(),
                )
            else:
                raise ValueError(f"Unsupported gate_mode: {gate_mode}")
            shared_dim = fusion_dim
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.shared_head = nn.Sequential(
            nn.Linear(shared_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden_channels, 2)
        self.group_head = nn.Linear(hidden_channels, 2)

    def encode_branches(
        self,
        stat_x: torch.Tensor,
        seq_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stat_embedding = self.stat_branch(stat_x)
        seq_embedding = self.seq_projection(self.seq_branch(seq_x))
        seq_embedding = self.apply_interaction(stat_embedding, seq_embedding)
        return stat_embedding, seq_embedding

    def apply_interaction(
        self,
        stat_embedding: torch.Tensor,
        seq_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.interaction_mode == "none":
            return seq_embedding
        if self.interaction_mode == "seq_gate":
            gate = self.interaction(stat_embedding)
            return gate * seq_embedding
        if self.interaction_mode == "seq_rescale":
            # Center the scale around 1.0 so sequence information is preserved
            # while still allowing light stat-guided suppression/amplification.
            scale = 0.5 + self.interaction(stat_embedding)
            return scale * seq_embedding
        gamma, beta = torch.chunk(self.interaction(stat_embedding), 2, dim=-1)
        gamma = torch.tanh(gamma)
        return seq_embedding * (1.0 + gamma) + beta

    def fuse_embeddings(
        self,
        stat_embedding: torch.Tensor,
        seq_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([stat_embedding, seq_embedding], dim=-1)
        stat_embedding = self.stat_fusion_projection(stat_embedding)
        seq_embedding = self.seq_fusion_projection(seq_embedding)
        gate_input = torch.cat([stat_embedding, seq_embedding], dim=-1)
        gate = self.gate(gate_input)
        if self.gate_mode == "scalar":
            return gate * stat_embedding + (1.0 - gate) * seq_embedding
        if self.gate_mode == "vector":
            return gate * stat_embedding + (1.0 - gate) * seq_embedding
        stat_gate, seq_gate = torch.chunk(gate, 2, dim=-1)
        return stat_gate * stat_embedding + seq_gate * seq_embedding

    def forward(
        self,
        stat_x: torch.Tensor,
        seq_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stat_embedding, seq_embedding = self.encode_branches(stat_x, seq_x)
        fused = self.fuse_embeddings(stat_embedding, seq_embedding)
        shared = self.shared_head(fused)
        return self.emotion_head(shared), self.group_head(shared)
