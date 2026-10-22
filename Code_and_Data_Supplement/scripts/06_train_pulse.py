"""Final PULSE privacy-refinement trainer.

The implementation follows the submitted objective. It expects cached anchors, identity basis,
and top-128 token rankings. Checkpoint selection is performed on the internal-validation set;
the selected deterministic inference point is evaluated separately by scripts/07_evaluate_nih.py.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pulse.attacker import PairwiseAttacker
from pulse.checkpoints import load_checkpoint, extract_state_dict
from pulse.config import load_config
from pulse.data import load_split, ImageDataset, SamePatientPairDataset
from pulse.losses import (UtilityWeights, kd_loss, embedding_distance, patient_confusion, pair_separation,
                          transfer_loss, variance_loss, subspace_loss)
from pulse.model import TeacherModel, PULSEStudent
from pulse.utils import set_seed, linear_schedule, autocast_context, cycle
from _common import build_transform, device

parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); cfg=load_config(args.config)
data,mc,tc=cfg["data"],cfg["model"],cfg["training"]; set_seed(int(tc["seed"])); dev=device(); transform=build_transform(cfg)
def split(name): return load_split(Path(data["split_dir"])/data["split_files"][name],data["image_root"],data["labels"],data["image_column"],data["patient_column"],data["path_column"])
retain,forget=split("retain"),split("forget")
payload=torch.load(cfg["assets"]["teacher_checkpoint"],map_location="cpu",weights_only=False); state=extract_state_dict(payload); n_patients=state["patient_head.weight"].shape[0]
teacher=TeacherModel(mc["backbone"],len(data["labels"]),n_patients,mc["embedding_dim"]).to(dev); load_checkpoint(teacher,cfg["assets"]["teacher_checkpoint"]); teacher.eval(); [p.requires_grad_(False) for p in teacher.parameters()]
student=PULSEStudent(mc["backbone"],len(data["labels"]),n_patients,mc["embedding_dim"],mc["bottleneck_dim"],mc["identity_subspace_dim"]).to(dev); student.load_state_dict(state,strict=False); student.set_identity_basis(torch.tensor(np.load(cfg["assets"]["identity_basis"]),dtype=torch.float32))
for p in student.parameters(): p.requires_grad_(False)
blocks=getattr(getattr(student.vision,"encoder",None),"layer",None) or getattr(getattr(student.vision,"encoder",None),"layers",None)
if blocks is None: raise RuntimeError("Could not locate Transformer blocks for this backbone")
for block in blocks[-int(mc["trainable_last_blocks"]):]:
    for p in block.parameters(): p.requires_grad_(True)
for module in (student.projection,student.bottleneck,student.disease_head):
    for p in module.parameters(): p.requires_grad_(True)
attacker=PairwiseAttacker(mc["embedding_dim"]).to(dev)
student_opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=float(tc["student_lr"]),weight_decay=float(tc["weight_decay"]))
attacker_opt=torch.optim.AdamW(attacker.parameters(),lr=float(tc["attacker_lr"]),weight_decay=float(tc["weight_decay"]))
scaler=torch.amp.GradScaler("cuda",enabled=bool(tc["mixed_precision"]) and torch.cuda.is_available())
ranking=np.load(cfg["assets"]["token_ranking"]); anchors=np.load(cfg["assets"]["anchors"])
retain_ds=ImageDataset(retain,transform,data["labels"],data["image_column"],data["patient_column"],data["path_column"])
forget_ds=ImageDataset(forget,transform,data["labels"],data["image_column"],data["patient_column"],data["path_column"],masks=ranking,anchors=anchors)
pair_ds=SamePatientPairDataset(forget_ds,seed=int(tc["seed"]))
retain_it=cycle(DataLoader(retain_ds,batch_size=int(tc["batch_size"]),shuffle=True,num_workers=2,drop_last=True))
forget_it=cycle(DataLoader(forget_ds,batch_size=int(tc["batch_size"]),shuffle=True,num_workers=2,drop_last=True))
pair_it=cycle(DataLoader(pair_ds,batch_size=int(tc["pair_batch_size"]),shuffle=True,num_workers=2,drop_last=True))
weights=UtilityWeights(**tc["utility_weights"]); schedules=tc["schedules"]; total=int(tc["epochs"])*int(tc["steps_per_epoch"]); output=Path(cfg["project"]["output_dir"]); output.mkdir(parents=True,exist_ok=True)
cardinalities=np.asarray(tc["token_cardinalities"]); probabilities=np.asarray(tc["token_probabilities"],dtype=float); probabilities/=probabilities.sum(); rng=np.random.default_rng(int(tc["seed"]))

def mask_for(batch, k): return batch["mask_indices"][:,:k].to(dev)
def bce(logits,labels): return F.binary_cross_entropy_with_logits(logits.float(),labels.float())
def attacker_binary(first,second,target): return F.binary_cross_entropy_with_logits(attacker(first,second).float(),torch.full((len(first),),float(target),device=dev))

global_step=0
for epoch in range(1,int(tc["epochs"])+1):
  student.train(); attacker.train()
  for _ in tqdm(range(int(tc["steps_per_epoch"])),desc=f"PULSE epoch {epoch}"):
    global_step+=1; progress=(global_step-1)/max(1,total-1); k=int(rng.choice(cardinalities,p=probabilities))
    alpha=linear_schedule(*schedules["alpha"],progress); beta=linear_schedule(*schedules["beta"],progress)
    lam={name:linear_schedule(*schedules[name],progress) for name in ("subspace","pair","patient","anchor","adversarial","transfer")}
    r,f,p=next(retain_it),next(forget_it),next(pair_it); ri=r["pixel_values"].to(dev); fi=f["pixel_values"].to(dev); rl=r["labels"].to(dev); fl=f["labels"].to(dev); fm=mask_for(f,k); fa=f["anchor"].to(dev)
    p1=p["first"]; p2=p["second"]; i1=p1["pixel_values"].to(dev); i2=p2["pixel_values"].to(dev); m1=mask_for(p1,k); m2=mask_for(p2,k)
    with torch.no_grad(), autocast_context(bool(tc["mixed_precision"])):
      tr=teacher(ri); tf=teacher(fi); tp1=teacher(i1); tp2=teacher(i2)
    # attacker updates on frozen Student embeddings
    for _a in range(int(tc["attacker_updates_per_student"])):
      attacker_opt.zero_grad(set_to_none=True)
      with torch.no_grad(), autocast_context(bool(tc["mixed_precision"])):
        c1=student(i1,alpha,beta); c2=student(i2,alpha,beta); mm1=student(i1,alpha,beta,m1); mm2=student(i2,alpha,beta,m2)
      neg=torch.roll(c2["embedding"],1,0)
      loss_a=.65*(attacker_binary(c1["embedding"],c2["embedding"],1)+attacker_binary(c1["embedding"],neg,0))/2 + .25*(attacker_binary(mm1["embedding"],mm2["embedding"],1)+attacker_binary(mm1["embedding"],torch.roll(mm2["embedding"],1,0),0))/2 + .10*(attacker_binary(c1["embedding"],mm2["embedding"],1)+attacker_binary(c1["embedding"],torch.roll(mm2["embedding"],1,0),0))/2
      loss_a.backward(); attacker_opt.step()
    for q in attacker.parameters(): q.requires_grad_(False)
    student_opt.zero_grad(set_to_none=True)
    with autocast_context(bool(tc["mixed_precision"])):
      sr=student(ri,alpha,beta); fc=student(fi,alpha,beta); fx=student(fi,alpha,beta,fm); c1=student(i1,alpha,beta); c2=student(i2,alpha,beta); x1=student(i1,alpha,beta,m1); x2=student(i2,alpha,beta,m2)
      util=weights.retain_bce*bce(sr["disease_logits"],rl)+weights.retain_kd*kd_loss(sr["disease_logits"],tr["disease_logits"])+weights.retain_embedding*embedding_distance(sr["embedding"],tr["embedding"])+weights.forget_clean_bce*bce(fc["disease_logits"],fl)+weights.forget_clean_kd*kd_loss(fc["disease_logits"],tf["disease_logits"])+weights.forget_masked_bce*bce(fx["disease_logits"],fl)+weights.forget_masked_kd*kd_loss(fx["disease_logits"],tf["disease_logits"])+weights.pair_clean_kd*(kd_loss(c1["disease_logits"],tp1["disease_logits"])+kd_loss(c2["disease_logits"],tp2["disease_logits"]))/2+weights.pair_masked_kd*(kd_loss(x1["disease_logits"],tp1["disease_logits"])+kd_loss(x2["disease_logits"],tp2["disease_logits"]))/2
      sub=.8*subspace_loss(fc["embedding"],student.identity_basis)+.2*subspace_loss(fx["embedding"],student.identity_basis)
      pid=.8*patient_confusion(fc["patient_logits"])+.2*patient_confusion(fx["patient_logits"])
      anc=(1-F.cosine_similarity(fx["embedding"].float(),fa.float(),dim=1)).mean()
      pair=.75*pair_separation(c1["embedding"],c2["embedding"],tc["pair_margin"])+.15*pair_separation(x1["embedding"],x2["embedding"],tc["pair_margin"])+.10*(pair_separation(c1["embedding"],x2["embedding"],tc["pair_margin"])+pair_separation(x1["embedding"],c2["embedding"],tc["pair_margin"]))/2
      adv=.80*attacker_binary(c1["embedding"],c2["embedding"],0)+.10*attacker_binary(x1["embedding"],x2["embedding"],0)+.05*attacker_binary(c1["embedding"],x2["embedding"],0)+.05*attacker_binary(x1["embedding"],c2["embedding"],0)
      transfer=.8*transfer_loss(fc["embedding"],fx["embedding"])+.2*(transfer_loss(c1["embedding"],x1["embedding"])+transfer_loss(c2["embedding"],x2["embedding"]))/2
      var=variance_loss(torch.cat([sr["embedding"],fc["embedding"],fx["embedding"],c1["embedding"],c2["embedding"]]),tc["variance_target"])
      loss=util+lam["anchor"]*anc+lam["subspace"]*sub+lam["patient"]*pid+lam["pair"]*pair+lam["adversarial"]*adv+lam["transfer"]*transfer+float(tc["variance_weight"])*var
    scaler.scale(loss).backward(); scaler.unscale_(student_opt); torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],float(tc["gradient_clip"])); scaler.step(student_opt); scaler.update()
    for q in attacker.parameters(): q.requires_grad_(True)
  torch.save({"epoch":epoch,"model_state_dict":student.state_dict(),"attacker_state_dict":attacker.state_dict(),"config":cfg},output/f"pulse_epoch{epoch:02d}.pt")

selected_source=output/"pulse_epoch04.pt"
selected_target=output/"pulse_selected.pt"
if selected_source.exists():
    import shutil
    shutil.copy2(selected_source,selected_target)
    print("Copied submitted epoch-4 selection to",selected_target)
print("Training complete.")
