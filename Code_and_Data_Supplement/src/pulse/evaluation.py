from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .metrics import macro_auroc
from .utils import autocast_context


@torch.inference_mode()
def evaluate_loader(model, loader, device: torch.device, alpha: float, beta: float,
                    collect_embeddings: bool = False, mixed_precision: bool = True) -> dict[str, Any]:
    labels, probabilities, embeddings, patient_ids = [], [], [], []
    model.eval()
    for batch in tqdm(loader, desc="evaluate"):
        images = batch["pixel_values"].to(device, non_blocking=True)
        with autocast_context(mixed_precision):
            output = model(images, alpha=alpha, beta=beta, mask_indices=None)
        labels.append(batch["labels"].numpy())
        probabilities.append(torch.sigmoid(output["disease_logits"].float()).cpu().numpy())
        if collect_embeddings:
            embeddings.append(F.normalize(output["embedding"].float(), dim=1).cpu().numpy().astype(np.float32))
            patient_ids.extend(str(value) for value in batch["patient_id"])
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    macro, per_label = macro_auroc(y_true, y_prob)
    result: dict[str, Any] = {"macro_auroc": macro, "per_label_auroc": per_label,
                              "labels": y_true, "probabilities": y_prob}
    if collect_embeddings:
        result["embeddings"] = np.concatenate(embeddings)
        result["patient_ids"] = np.asarray(patient_ids, dtype=str)
    return result
