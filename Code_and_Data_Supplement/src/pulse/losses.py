from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(student_logits.float(), torch.sigmoid(teacher_logits.detach().float()))


def embedding_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(first.float(), second.detach().float(), dim=1)).mean()


def patient_confusion(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits.float(), dim=1)
    confidence = probabilities.max(dim=1).values
    entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1) / math.log(probabilities.shape[1])
    return (confidence + 1.0 - entropy).mean()


def pair_separation(first: torch.Tensor, second: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    similarity = F.cosine_similarity(first.float(), second.float(), dim=1)
    return F.relu(similarity - margin).mean()


def transfer_loss(clean: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(clean.float(), masked.detach().float(), dim=1)).mean()


def variance_loss(embedding: torch.Tensor, target: float = 0.045) -> torch.Tensor:
    standard_deviation = embedding.float().std(dim=0, unbiased=False)
    return F.relu(float(target) - standard_deviation).mean()


def subspace_loss(embedding: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    embedding = F.normalize(embedding.float(), dim=1)
    coefficients = embedding @ basis.to(embedding)
    return coefficients.square().sum(dim=1).mean()


@dataclass(frozen=True)
class UtilityWeights:
    retain_bce: float = 0.5
    retain_kd: float = 4.0
    retain_embedding: float = 0.6
    forget_clean_bce: float = 1.0
    forget_clean_kd: float = 6.0
    forget_masked_bce: float = 0.5
    forget_masked_kd: float = 3.0
    pair_clean_kd: float = 0.7
    pair_masked_kd: float = 0.3
