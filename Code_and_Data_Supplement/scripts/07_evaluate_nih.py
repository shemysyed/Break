from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from pulse.checkpoints import extract_state_dict, load_checkpoint
from pulse.config import load_config
from pulse.data import load_split, ImageDataset
from pulse.evaluation import evaluate_loader
from pulse.metrics import pair_reid_auc, retrieval_metrics, drop_r
from pulse.model import PULSEStudent
from _common import build_transform, device

parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--output",default=None); args=parser.parse_args(); cfg=load_config(args.config); data,mc,ev=cfg["data"],cfg["model"],cfg["evaluation"]; dev=device(); transform=build_transform(cfg)
payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False); state=extract_state_dict(payload); n_patients=state["patient_head.weight"].shape[0]
model=PULSEStudent(mc["backbone"],len(data["labels"]),n_patients,mc["embedding_dim"],mc["bottleneck_dim"],mc["identity_subspace_dim"]).to(dev); load_checkpoint(model,args.checkpoint,strict=True); model.eval()
def split(name): return load_split(Path(data["split_dir"])/data["split_files"][name],data["image_root"],data["labels"],data["image_column"],data["patient_column"],data.get("path_column","ImagePath"),data.get("label_column"))
def loader(df): return DataLoader(ImageDataset(df,transform,data["labels"],data["image_column"],data["patient_column"],data.get("path_column","ImagePath")),batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
retain,forget=split("retain"),split("forget"); alpha=float(cfg["inference"]["alpha"]); beta=float(cfg["inference"]["beta"])
r=evaluate_loader(model,loader(retain),dev,alpha,beta,False); f=evaluate_loader(model,loader(forget),dev,alpha,beta,True)
pair=pair_reid_auc(f["embeddings"],f["patient_ids"],int(ev["pair_count"]),int(ev["pair_seed"])); retrieval=retrieval_metrics(f["embeddings"],f["patient_ids"],int(ev["retrieval_chunk_size"])); result={"retain_macro_auroc":r["macro_auroc"],"forget_macro_auroc":f["macro_auroc"],"pair_reid_auc":pair,**retrieval,"drop_r":drop_r(retrieval["map_at_r"],float(ev["teacher_map_at_r"])),"alpha":alpha,"beta":beta,"pair_score_direction":"raw cosine; no inversion"}
print(json.dumps(result,indent=2)); output=Path(args.output or Path(args.checkpoint).with_suffix(".metrics.json")); output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print("Saved",output)
