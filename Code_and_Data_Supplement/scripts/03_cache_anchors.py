from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from pulse.config import load_config
from pulse.data import load_split, ImageDataset
from pulse.model import TeacherModel
from pulse.checkpoints import load_checkpoint
from pulse.utils import set_seed, autocast_context
from _common import build_transform, device

parser=argparse.ArgumentParser(); parser.add_argument("--config", required=True); args=parser.parse_args(); cfg=load_config(args.config)
data, model_cfg, anchor_cfg = cfg["data"], cfg["model"], cfg["anchor"]; set_seed(int(anchor_cfg["seed"])); dev=device(); transform=build_transform(cfg)
retain=load_split(Path(data["split_dir"])/data["split_files"]["retain"], data["image_root"], data["labels"], data["image_column"], data["patient_column"], data["path_column"])
forget=load_split(Path(data["split_dir"])/data["split_files"]["forget"], data["image_root"], data["labels"], data["image_column"], data["patient_column"], data["path_column"])
checkpoint=torch.load(cfg["assets"]["teacher_checkpoint"], map_location="cpu", weights_only=False); state=checkpoint["model_state_dict"]; n_patients=state["patient_head.weight"].shape[0]
model=TeacherModel(model_cfg["backbone"], len(data["labels"]), n_patients, model_cfg["embedding_dim"]).to(dev); load_checkpoint(model, cfg["assets"]["teacher_checkpoint"]); model.eval()
@torch.inference_mode()
def extract(df):
    loader=DataLoader(ImageDataset(df, transform, data["labels"], data["image_column"], data["patient_column"], data["path_column"]), batch_size=64, shuffle=False, num_workers=4)
    probabilities=[]; embeddings=[]
    for batch in loader:
        with autocast_context(True): output=model(batch["pixel_values"].to(dev))
        probabilities.append(torch.sigmoid(output["disease_logits"].float()).cpu().numpy()); embeddings.append(F.normalize(output["embedding"].float(),dim=1).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(embeddings)
rng=np.random.default_rng(int(anchor_cfg["seed"])); pool_size=min(int(anchor_cfg["reference_pool_size"]),len(retain)); pool_index=rng.choice(len(retain),size=pool_size,replace=False)
retain_prob, retain_emb=extract(retain.iloc[pool_index].reset_index(drop=True)); forget_prob,_=extract(forget)
neighbors=NearestNeighbors(n_neighbors=int(anchor_cfg["neighbors"]), metric="euclidean").fit(retain_prob); indices=neighbors.kneighbors(forget_prob, return_distance=False)
anchors=retain_emb[indices].mean(axis=1); anchors/=np.clip(np.linalg.norm(anchors,axis=1,keepdims=True),1e-8,None)
path=Path(cfg["assets"]["anchors"]); path.parent.mkdir(parents=True,exist_ok=True); np.save(path,anchors.astype(np.float32)); print("Saved",path,anchors.shape)
