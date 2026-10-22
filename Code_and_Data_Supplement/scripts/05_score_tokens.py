from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pulse.config import load_config
from pulse.data import load_split, ImageDataset
from pulse.model import TeacherModel
from pulse.checkpoints import load_checkpoint, extract_state_dict
from pulse.token_scoring import six_effect_scores
from _common import build_transform, device

parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); cfg=load_config(args.config); data=cfg["data"]; dev=device(); transform=build_transform(cfg)
forget=load_split(Path(data["split_dir"])/data["split_files"]["forget"],data["image_root"],data["labels"],data["image_column"],data["patient_column"],data["path_column"])
payload=torch.load(cfg["assets"]["teacher_checkpoint"],map_location="cpu",weights_only=False); state=extract_state_dict(payload); n_patients=state["patient_head.weight"].shape[0]
model=TeacherModel(cfg["model"]["backbone"],len(data["labels"]),n_patients,cfg["model"]["embedding_dim"]).to(dev); load_checkpoint(model,cfg["assets"]["teacher_checkpoint"]); model.eval()
with open(cfg["assets"]["patient_mapping"],"r",encoding="utf-8") as handle: mapping=json.load(handle)
anchors=np.load(cfg["assets"]["anchors"]); basis=torch.tensor(np.load(cfg["assets"]["identity_basis"]),device=dev)
loader=DataLoader(ImageDataset(forget,transform,data["labels"],data["image_column"],data["patient_column"],data["path_column"],anchors=anchors),batch_size=int(cfg["token_scoring"]["batch_size"]),shuffle=False,num_workers=2)
ranking=[]
for batch in tqdm(loader,desc="six-effect token scoring"):
    patient=torch.tensor([mapping[str(value)] for value in batch["patient_id"]],device=dev)
    scores,_=six_effect_scores(model,batch["pixel_values"].to(dev),patient,batch["anchor"].to(dev),basis,float(cfg["token_scoring"]["lambda_token"]))
    ranking.append(torch.argsort(scores,dim=1,descending=True)[:,:int(cfg["token_scoring"]["ranking_size"])].cpu().numpy().astype(np.int16))
ranking=np.concatenate(ranking); path=Path(cfg["assets"]["token_ranking"]); path.parent.mkdir(parents=True,exist_ok=True); np.save(path,ranking); print("Saved",path,ranking.shape)
for k in cfg["training"]["token_cardinalities"]: np.save(path.with_name(f"token_masks_k{k}.npy"),ranking[:,:int(k)])
