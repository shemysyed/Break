from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseAttacker(nn.Module):
    """Pairwise linkage attacker with the 4d+4 feature vector from the paper."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        input_dim = 4 * embedding_dim + 4
        self.network = nn.Sequential(
            nn.Linear(input_dim, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(768, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 1),
        )

    @staticmethod
    def features(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = F.normalize(first.float(), dim=1)
        second = F.normalize(second.float(), dim=1)
        delta = first - second
        cosine = (first * second).sum(dim=1, keepdim=True)
        l1 = delta.abs().mean(dim=1, keepdim=True)
        l2 = delta.norm(dim=1, keepdim=True)
        squared = delta.square().mean(dim=1, keepdim=True)
        return torch.cat([first, second, delta.abs(), first * second, cosine, l1, l2, squared], dim=1)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(first, second)).squeeze(1)
