from __future__ import annotations
import torch


def identity_basis_from_patient_head(weight: torch.Tensor, dimension: int = 96) -> torch.Tensor:
    """Return the leading right-singular directions of the centered patient head."""
    if weight.ndim != 2:
        raise ValueError("Patient-head weight must have shape [patients, embedding_dim]")
    centered = weight.float() - weight.float().mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    if dimension > vh.shape[0]:
        raise ValueError(f"Requested q={dimension}, but only {vh.shape[0]} directions are available")
    return vh[:dimension].T.contiguous()


def subspace_energy(embedding: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
    coefficients = embedding @ basis.to(embedding)
    return coefficients.square().sum(dim=-1)
