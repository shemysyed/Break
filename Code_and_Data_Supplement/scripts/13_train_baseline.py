"""Shared controlled baseline-adaptation entry point.

This script centralizes initialization, patient-disjoint data loading, trainable components,
and optimization budget. Method-specific objectives are deliberately explicit below so the
controlled adaptations can be inspected in one file.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pulse.checkpoints import load_checkpoint
from pulse.config import load_config
from pulse.data import load_split, ImageDataset
from pulse.losses import kd_loss, patient_confusion
from pulse.model import TeacherModel
from pulse.utils import set_seed, autocast_context, cycle
from _common import build_transform, device

METHODS={"standard_retrain","retain_ft","neggrad_plus","forget_mi","bad_teaching","scrub","salun","privacy_retrain"}
parser=argparse.ArgumentParser(); parser.add_argument("--paper-config",default="configs/nih_pulse.yaml"); parser.add_argument("--baseline-config",default="configs/baselines.yaml"); parser.add_argument("--method",required=True,choices=sorted(METHODS)); parser.add_argument("--teacher",default="checkpoints/teacher_nih.pt"); args=parser.parse_args()
paper=load_config(args.paper_config); baseline=load_config(args.baseline_config); shared=baseline["shared"]; spec=baseline["methods"][args.method]; data=paper["data"]; mc=paper["model"]; set_seed(int(shared["seed"])); dev=device(); transform=build_transform(paper)
def split(name): return load_split(Path(data["split_dir"])/data["split_files"][name],data["image_root"],data["labels"],data["image_column"],data["patient_column"],data.get("path_column","ImagePath"),data.get("label_column"))
retain,forget=split("retain"),split("forget")
teacher_payload=torch.load(args.teacher,map_location="cpu",weights_only=False); teacher_state=teacher_payload.get("model_state_dict",teacher_payload); n_patients=teacher_state["patient_head.weight"].shape[0]
teacher=TeacherModel(mc["backbone"],len(data["labels"]),n_patients,mc["embedding_dim"]).to(dev); load_checkpoint(teacher,args.teacher); teacher.eval(); [p.requires_grad_(False) for p in teacher.parameters()]
model=TeacherModel(mc["backbone"],len(data["labels"]),n_patients,mc["embedding_dim"]).to(dev)
if spec["initialization"]=="teacher": model.load_state_dict(teacher.state_dict())
for p in model.parameters(): p.requires_grad_(False)
blocks=getattr(getattr(model.vision,"encoder",None),"layer",None) or getattr(getattr(model.vision,"encoder",None),"layers",None)
if blocks is None: raise RuntimeError("Could not locate Transformer blocks")
for block in blocks[-int(shared["trainable_last_blocks"]):]:
    for p in block.parameters(): p.requires_grad_(True)
for module in (model.projection,model.disease_head):
    for p in module.parameters(): p.requires_grad_(True)
retain_it=cycle(DataLoader(ImageDataset(retain,transform,data["labels"],data["image_column"],data["patient_column"],data.get("path_column","ImagePath")),batch_size=int(shared["batch_size"]),shuffle=True,num_workers=2,drop_last=True))
forget_it=cycle(DataLoader(ImageDataset(forget,transform,data["labels"],data["image_column"],data["patient_column"],data.get("path_column","ImagePath")),batch_size=int(shared["batch_size"]),shuffle=True,num_workers=2,drop_last=True))
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(shared["learning_rate"]),weight_decay=float(shared["weight_decay"])); output=Path("outputs/baselines")/args.method; output.mkdir(parents=True,exist_ok=True)
saliency_masks=None
for epoch in range(1,int(shared["epochs"])+1):
  model.train()
  for _ in tqdm(range(int(shared["steps_per_epoch"])),desc=f"{args.method} epoch {epoch}"):
    r=next(retain_it); f=next(forget_it); ri=r["pixel_values"].to(dev); fi=f["pixel_values"].to(dev); rl=r["labels"].to(dev)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(), autocast_context(True): tr=teacher(ri); tf=teacher(fi)
    with autocast_context(True):
      mr=model(ri); mf=model(fi); retain_loss=float(shared["retain_bce_weight"])*F.binary_cross_entropy_with_logits(mr["disease_logits"].float(),rl.float())+float(shared["retain_kd_weight"])*kd_loss(mr["disease_logits"],tr["disease_logits"])
      if args.method in {"standard_retrain","retain_ft"}: loss=retain_loss
      elif args.method=="neggrad_plus": loss=retain_loss-float(shared["forget_weight"])*kd_loss(mf["disease_logits"],tf["disease_logits"])
      elif args.method=="forget_mi": loss=retain_loss+float(shared["forget_weight"])*(mf["embedding"].float().var(dim=0).mean()+patient_confusion(mf["patient_logits"]))
      elif args.method=="bad_teaching": shuffled=torch.roll(tf["disease_logits"],1,0); loss=retain_loss+float(shared["forget_weight"])*kd_loss(mf["disease_logits"],shuffled)
      elif args.method=="scrub": loss=retain_loss-float(shared["forget_weight"])*kd_loss(mf["disease_logits"],tf["disease_logits"])
      elif args.method=="salun": loss=retain_loss-float(shared["forget_weight"])*kd_loss(mf["disease_logits"],tf["disease_logits"])
      elif args.method=="privacy_retrain": normalized=F.normalize(mr["embedding"].float(),dim=1); loss=retain_loss+float(shared["verifier_weight"])*(normalized@normalized.T).triu(1).square().mean()
    loss.backward()
    if args.method=="salun":
      fraction=float(spec["saliency_fraction"])
      all_values=torch.cat([p.grad.detach().abs().flatten() for p in model.parameters() if p.grad is not None]); threshold=torch.quantile(all_values,1-fraction)
      for p in model.parameters():
        if p.grad is not None: p.grad.mul_((p.grad.abs()>=threshold).to(p.grad))
    torch.nn.utils.clip_grad_norm_(model.parameters(),float(shared["gradient_clip"])); optimizer.step()
  torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"method":args.method,"paper_config":paper,"baseline_config":baseline},output/f"{args.method}_epoch{epoch:02d}.pt")
print("Saved baseline checkpoints to",output)
