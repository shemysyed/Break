from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .identity import subspace_energy


@dataclass
class TokenEffects:
    patient_logit: torch.Tensor
    subspace: torch.Tensor
    anchor: torch.Tensor
    disease_probability: torch.Tensor
    disease_logit: torch.Tensor
    embedding: torch.Tensor


def rank_normalize(values: torch.Tensor) -> torch.Tensor:
    """Within-row ranks in [0,1], with larger values receiving larger ranks."""
    if values.ndim != 2:
        raise ValueError("rank_normalize expects [batch,tokens]")
    order = torch.argsort(values, dim=1, stable=True)
    ranks = torch.argsort(order, dim=1, stable=True).float()
    return ranks / max(1, values.shape[1] - 1)


@torch.inference_mode()
def six_effect_scores(
    teacher,
    pixel_values: torch.Tensor,
    patient_indices: torch.Tensor,
    anchors: torch.Tensor,
    identity_basis: torch.Tensor,
    lambda_token: float = 0.75,
) -> tuple[torch.Tensor, TokenEffects]:
    """Compute the submitted six-effect disease-protected token score."""
    clean = teacher(pixel_values)
    tokens = clean["tokens"]
    patch_count = tokens.shape[1] - 1
    clean_embedding = F.normalize(clean["embedding"].float(), dim=1)
    anchors = F.normalize(anchors.float(), dim=1)
    clean_probability = torch.sigmoid(clean["disease_logits"].float())
    clean_subspace = subspace_energy(clean_embedding, identity_basis)
    clean_anchor = (clean_embedding * anchors).sum(dim=1)
    clean_patient = clean["patient_logits"].gather(1, patient_indices[:, None]).squeeze(1)

    outputs = [torch.empty(tokens.shape[0], patch_count, device=tokens.device) for _ in range(6)]
    for token_index in range(patch_count):
        index = torch.full((tokens.shape[0], 1), token_index, device=tokens.device, dtype=torch.long)
        changed_tokens = torch.cat([tokens[:, :1], tokens[:, 1:].clone()], dim=1)
        patches = changed_tokens[:, 1:]
        patches[torch.arange(tokens.shape[0], device=tokens.device), token_index] = patches.mean(dim=1)
        changed = teacher.from_tokens(changed_tokens)
        changed_embedding = F.normalize(changed["embedding"].float(), dim=1)
        changed_patient = changed["patient_logits"].gather(1, patient_indices[:, None]).squeeze(1)
        outputs[0][:, token_index] = clean_patient - changed_patient
        outputs[1][:, token_index] = clean_subspace - subspace_energy(changed_embedding, identity_basis)
        outputs[2][:, token_index] = (changed_embedding * anchors).sum(dim=1) - clean_anchor
        outputs[3][:, token_index] = (torch.sigmoid(changed["disease_logits"].float()) - clean_probability).abs().mean(dim=1)
        outputs[4][:, token_index] = (changed["disease_logits"].float() - clean["disease_logits"].float()).abs().mean(dim=1)
        outputs[5][:, token_index] = 1.0 - (clean_embedding * changed_embedding).sum(dim=1)

    effects = TokenEffects(*outputs)
    identity_score = (rank_normalize(outputs[0]) + rank_normalize(outputs[1]) + rank_normalize(outputs[2])) / 3.0
    protection = (rank_normalize(outputs[3]) + rank_normalize(outputs[4]) + rank_normalize(outputs[5])) / 3.0
    return identity_score - float(lambda_token) * protection, effects
