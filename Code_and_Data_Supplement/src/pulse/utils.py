from __future__ import annotations
import contextlib
import json
import random
from pathlib import Path
from typing import Iterator
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def linear_schedule(start: float, end: float, progress: float) -> float:
    progress = min(1.0, max(0.0, float(progress)))
    return float((1.0 - progress) * start + progress * end)


@contextlib.contextmanager
def autocast_context(enabled: bool = True) -> Iterator[None]:
    if enabled and torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def save_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def cycle(loader):
    while True:
        yield from loader
