from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import ChebConv, GATConv, GCNConv, DenseGCNConv


def build_batched_edge_index(
    edge_index: torch.Tensor,
    num_graphs: int,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    edge_index = edge_index.to(device)
    num_edges = edge_index.size(1)
    offsets = torch.arange(num_graphs, device=device).repeat_interleave(num_edges) * num_nodes
    return edge_index.repeat(1, num_graphs) + offsets.unsqueeze(0)


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class GraphBackbone(nn.Module):
    def __init__(
        self,
        conv_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_nodes: int,
        dropout: float,
        cheb_k: int = 3,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)
        self.hidden_channels = hidden_channels

        if conv_type == "adaptive_gcn":
            self.input_conv = DenseGCNConv(in_channels, hidden_channels)
            self.hidden_convs = nn.ModuleList(
                [DenseGCNConv(hidden_channels, hidden_channels) for _ in range(max(num_layers - 1, 0))]
            )
            self.base_adj_logits = nn.Parameter(torch.eye(num_nodes))
        else:
            conv_cls = {
                "gcn": lambda c_in, c_out: GCNConv(c_in, c_out),
                "cheb": lambda c_in, c_out: ChebConv(c_in, c_out, K=cheb_k),
                "gat": lambda c_in, c_out: GATConv(c_in, c_out // gat_heads, heads=gat_heads, concat=True),
            }[conv_type]
            self.input_conv = conv_cls(in_channels, hidden_channels)
            self.hidden_convs = nn.ModuleList(
                [conv_cls(hidden_channels, hidden_channels) for _ in range(max(num_layers - 1, 0))]
            )
        self.input_norm = nn.LayerNorm(hidden_channels)
        self.hidden_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(max(num_layers - 1, 0))]
        )

    def adaptive_adj(self, base_adj: torch.Tensor) -> torch.Tensor:
        learned = torch.sigmoid(self.base_adj_logits)
        learned = 0.5 * (learned + learned.transpose(0, 1))
        adj = torch.maximum(base_adj, learned)
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj / degree

    def forward(
        self,
        x: torch.Tensor,
        base_adj: torch.Tensor,
        base_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, W, N, F]
        batch_size, num_windows, num_nodes, in_channels = x.shape
        if self.conv_type == "adaptive_gcn":
            flat = x.reshape(batch_size * num_windows, num_nodes, in_channels)
            adj = self.adaptive_adj(base_adj)
            out = self.input_conv(flat, adj)
            out = self.input_norm(out)
            out = torch.relu(out)
            out = self.dropout(out)
            for conv, norm in zip(self.hidden_convs, self.hidden_norms, strict=True):
                residual = out
                out = conv(out, adj)
                out = norm(out)
                out = torch.relu(out)
                out = self.dropout(out)
                out = out + residual
            return out.reshape(batch_size, num_windows, num_nodes, self.hidden_channels)

        num_graphs = batch_size * num_windows
        flat_nodes = x.reshape(num_graphs * num_nodes, in_channels)
        edge_index = build_batched_edge_index(base_edge_index, num_graphs, num_nodes, x.device)
        out = self.input_conv(flat_nodes, edge_index)
        out = out.reshape(num_graphs, num_nodes, self.hidden_channels)
        out = self.input_norm(out)
        out = torch.relu(out)
        out = self.dropout(out)

        for conv, norm in zip(self.hidden_convs, self.hidden_norms, strict=True):
            residual = out
            flat = out.reshape(num_graphs * num_nodes, self.hidden_channels)
            flat = conv(flat, edge_index)
            out = flat.reshape(num_graphs, num_nodes, self.hidden_channels)
            out = norm(out)
            out = torch.relu(out)
            out = self.dropout(out)
            out = out + residual
        return out.reshape(batch_size, num_windows, num_nodes, self.hidden_channels)


class GraphSequenceClassifier(nn.Module):
    def __init__(
        self,
        conv_type: str,
        temporal_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_nodes: int,
        dropout: float,
        cheb_k: int = 3,
        gat_heads: int = 4,
        temporal_heads: int = 4,
    ):
        super().__init__()
        self.temporal_type = temporal_type
        self.backbone = GraphBackbone(
            conv_type=conv_type,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_nodes=num_nodes,
            dropout=dropout,
            cheb_k=cheb_k,
            gat_heads=gat_heads,
        )
        if temporal_type == "attention":
            self.temporal_attention = nn.Linear(hidden_channels, 1)
            temporal_dim = hidden_channels
        elif temporal_type == "mean":
            temporal_dim = hidden_channels
        elif temporal_type == "meanstd":
            temporal_dim = hidden_channels * 2
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

    def encode_sequence(
        self,
        x: torch.Tensor,
        base_adj: torch.Tensor,
        base_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.backbone(x, base_adj, base_edge_index)
        return x.mean(dim=2)  # [B, W, H]

    def pool_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_type == "mean":
            return x.mean(dim=1)
        if self.temporal_type == "attention":
            weights = torch.softmax(self.temporal_attention(x).squeeze(-1), dim=1)
            return torch.sum(x * weights.unsqueeze(-1), dim=1)
        if self.temporal_type == "meanstd":
            mean = x.mean(dim=1)
            std = x.std(dim=1, unbiased=False)
            return torch.cat([mean, std], dim=-1)
        if self.temporal_type == "gru":
            _, hidden = self.temporal_gru(x)
            return hidden[-1]
        if self.temporal_type == "selfattn":
            x = self.temporal_encoder(x)
            weights = torch.softmax(self.temporal_attention(x).squeeze(-1), dim=1)
            return torch.sum(x * weights.unsqueeze(-1), dim=1)
        raise RuntimeError("Unsupported temporal mode reached")

    def forward(
        self,
        x: torch.Tensor,
        base_adj: torch.Tensor,
        base_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self.encode_sequence(x, base_adj, base_edge_index)
        pooled = self.pool_sequence(sequence)
        return self.classifier(pooled)
