from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from transformers import AutoModel


def _mean_pool_patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] < 2:
        raise ValueError(f"Expected [B,1+M,H] tokens, received {tuple(tokens.shape)}")
    return tokens[:, 1:, :].mean(dim=1)


class TeacherModel(nn.Module):
    def __init__(self, backbone: str, n_classes: int, n_patients: int, embedding_dim: int = 256,
                 local_files_only: bool = False) -> None:
        super().__init__()
        self.vision = AutoModel.from_pretrained(backbone, local_files_only=local_files_only)
        hidden = int(self.vision.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU()
        )
        self.disease_head = nn.Linear(embedding_dim, n_classes)
        self.patient_head = nn.Linear(embedding_dim, n_patients)

    def encode_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision(pixel_values=pixel_values, return_dict=True).last_hidden_state

    def from_tokens(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.projection(_mean_pool_patch_tokens(tokens))
        return {
            "embedding": embedding,
            "disease_logits": self.disease_head(embedding),
            "patient_logits": self.patient_head(embedding),
            "tokens": tokens,
        }

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.from_tokens(self.encode_tokens(pixel_values))


class PULSEStudent(TeacherModel):
    def __init__(self, backbone: str, n_classes: int, n_patients: int, embedding_dim: int = 256,
                 bottleneck_dim: int = 32, identity_subspace_dim: int = 96,
                 local_files_only: bool = False) -> None:
        super().__init__(backbone, n_classes, n_patients, embedding_dim, local_files_only)
        self.bottleneck = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, embedding_dim)
        )
        self.register_buffer("identity_basis", torch.zeros(embedding_dim, identity_subspace_dim))
        self.register_buffer("default_alpha", torch.tensor(0.0))
        self.register_buffer("default_beta", torch.tensor(0.0))

    def set_identity_basis(self, basis: torch.Tensor) -> None:
        if basis.shape != self.identity_basis.shape:
            raise ValueError(f"Expected basis {tuple(self.identity_basis.shape)}, got {tuple(basis.shape)}")
        self.identity_basis.copy_(basis.to(self.identity_basis))

    @staticmethod
    def intervene(tokens: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor:
        if indices is None:
            return tokens
        cls_token, patch_tokens = tokens[:, :1, :], tokens[:, 1:, :]
        batch, count, hidden = patch_tokens.shape
        if indices.ndim != 2 or indices.shape[0] != batch:
            raise ValueError("indices must have shape [batch,K]")
        mask = torch.zeros(batch, count, 1, device=tokens.device, dtype=patch_tokens.dtype)
        batch_index = torch.arange(batch, device=tokens.device)[:, None].expand_as(indices)
        mask[batch_index, indices.long(), 0] = 1.0
        mean_token = patch_tokens.mean(dim=1, keepdim=True).expand(batch, count, hidden)
        return torch.cat([cls_token, patch_tokens * (1.0 - mask) + mean_token * mask], dim=1)

    def privacy_transform(self, embedding: torch.Tensor, alpha: float | torch.Tensor | None,
                          beta: float | torch.Tensor | None) -> torch.Tensor:
        alpha_t = self.default_alpha if alpha is None else torch.as_tensor(alpha, device=embedding.device)
        beta_t = self.default_beta if beta is None else torch.as_tensor(beta, device=embedding.device)
        alpha_t = alpha_t.to(dtype=embedding.dtype)
        beta_t = beta_t.to(dtype=embedding.dtype)
        h = (1.0 - alpha_t) * embedding + alpha_t * self.bottleneck(embedding)
        basis = self.identity_basis.to(device=h.device, dtype=h.dtype)
        return h - beta_t * ((h @ basis) @ basis.T)

    def branch_from_tokens(self, tokens: torch.Tensor, alpha: float | None, beta: float | None,
                           mask_indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self.intervene(tokens, mask_indices)
        base_embedding = self.projection(_mean_pool_patch_tokens(tokens))
        embedding = self.privacy_transform(base_embedding, alpha, beta)
        return {
            "embedding": embedding,
            "base_embedding": base_embedding,
            "disease_logits": self.disease_head(embedding),
            "patient_logits": self.patient_head(embedding),
            "tokens": tokens,
        }

    def forward(self, pixel_values: torch.Tensor, alpha: float | None = None, beta: float | None = None,
                mask_indices: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self.encode_tokens(pixel_values)
        return self.branch_from_tokens(tokens, alpha, beta, mask_indices)
