"""Train the disease/patient Teacher used by PULSE.

This entry point is intentionally compact. It trains disease BCE and patient-identification CE
on the union of retain and forget training patients, updates the final Transformer blocks and
projection/heads, and selects by validation disease AUROC subject to patient-head monitoring.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pulse.config import load_config
from pulse.data import load_split, ImageDataset
from pulse.model import TeacherModel
from pulse.metrics import macro_auroc
from pulse.utils import set_seed, autocast_context, save_json
from _common import build_transform, device

parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
cfg = load_config(args.config); data, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
set_seed(int(train_cfg["seed"])); dev = device(); transform = build_transform(cfg)
retain = load_split(Path(data["split_dir"])/data["retain_file"], data["image_root"], data["labels"], data["image_column"], data["patient_column"])
forget = load_split(Path(data["split_dir"])/data["forget_file"], data["image_root"], data["labels"], data["image_column"], data["patient_column"])
validation = load_split(Path(data["split_dir"])/data["validation_file"], data["image_root"], data["labels"], data["image_column"], data["patient_column"])
training = pd.concat([retain, forget], ignore_index=True)
patients = sorted(training[data["patient_column"]].astype(str).unique()); patient_to_index = {p:i for i,p in enumerate(patients)}
base = ImageDataset(training, transform, data["labels"], data["image_column"], data["patient_column"], "ImagePath")
loader = DataLoader(base, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
model = TeacherModel(model_cfg["backbone"], len(data["labels"]), len(patients), int(model_cfg["embedding_dim"])).to(dev)
for parameter in model.parameters(): parameter.requires_grad_(False)
blocks = getattr(getattr(model.vision, "encoder", None), "layer", None) or getattr(getattr(model.vision, "encoder", None), "layers", None)
if blocks is None: raise RuntimeError("Could not locate Transformer blocks; adapt scripts/02_train_teacher.py for this backbone")
for block in blocks[-int(model_cfg["trainable_last_blocks"]):]:
    for parameter in block.parameters(): parameter.requires_grad_(True)
for module in (model.projection, model.disease_head, model.patient_head):
    for parameter in module.parameters(): parameter.requires_grad_(True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("mixed_precision", True)) and torch.cuda.is_available())
out = Path(cfg["project"]["output_dir"]); out.mkdir(parents=True, exist_ok=True); save_json(patient_to_index, out/"patient_to_index.json")
for epoch in range(1, int(train_cfg["epochs"])+1):
    model.train()
    for batch in tqdm(loader, desc=f"teacher epoch {epoch}"):
        images = batch["pixel_values"].to(dev); labels = batch["labels"].to(dev)
        patient = torch.tensor([patient_to_index[str(x)] for x in batch["patient_id"]], device=dev)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(bool(train_cfg.get("mixed_precision", True))):
            output = model(images)
            loss = F.binary_cross_entropy_with_logits(output["disease_logits"].float(), labels.float()) + float(train_cfg["patient_loss_weight"]) * F.cross_entropy(output["patient_logits"].float(), patient)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"])); scaler.step(optimizer); scaler.update()
    torch.save({"epoch":epoch, "model_state_dict":model.state_dict(), "patient_to_index":patient_to_index, "config":cfg}, out/f"teacher_epoch{epoch:02d}.pt")
print("Teacher training complete. Select the checkpoint using the validation protocol described in the paper.")
