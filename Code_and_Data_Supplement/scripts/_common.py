from __future__ import annotations
from pathlib import Path
import torch
from torchvision import transforms
from transformers import AutoImageProcessor
from pulse.config import load_config


def build_transform(config: dict):
    model = config["model"]
    processor = AutoImageProcessor.from_pretrained(
        model["backbone"], local_files_only=bool(model.get("local_files_only", False))
    )
    return transforms.Compose([
        transforms.Resize((int(model.get("input_size", 224)), int(model.get("input_size", 224)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
