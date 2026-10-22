from __future__ import annotations
from pathlib import Path
from typing import Any
import torch


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "student_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
        else:
            state = payload
    else:
        state = payload
    return {str(key).removeprefix("module.").removeprefix("_orig_mod."): value for key, value in state.items()}


def load_checkpoint(model: torch.nn.Module, path: str | Path, strict: bool = True) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(extract_state_dict(payload), strict=strict)
    return payload if isinstance(payload, dict) else {}
