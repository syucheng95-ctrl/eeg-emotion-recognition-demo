from __future__ import annotations

import torch
from torch import nn

from .multitask_models import MultiTaskDualBranchClassifier


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_adv)


class AdversarialDualBranchClassifier(MultiTaskDualBranchClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        stat_x: torch.Tensor,
        seq_x: torch.Tensor,
        lambda_adv: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stat_embedding, seq_embedding = self.encode_branches(stat_x, seq_x)
        fused = self.fuse_embeddings(stat_embedding, seq_embedding)
        shared = self.shared_head(fused)
        emotion_logits = self.emotion_head(shared)
        group_logits = self.group_head(grad_reverse(shared, lambda_adv))
        return emotion_logits, group_logits
