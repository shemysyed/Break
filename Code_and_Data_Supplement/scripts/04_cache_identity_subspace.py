from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from pulse.config import load_config
from pulse.checkpoints import extract_state_dict
from pulse.identity import identity_basis_from_patient_head

parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); cfg=load_config(args.config)
payload=torch.load(cfg["assets"]["teacher_checkpoint"],map_location="cpu",weights_only=False); state=extract_state_dict(payload)
basis=identity_basis_from_patient_head(state["patient_head.weight"],int(cfg["model"]["identity_subspace_dim"]))
path=Path(cfg["assets"]["identity_basis"]); path.parent.mkdir(parents=True,exist_ok=True); np.save(path,basis.cpu().numpy().astype(np.float32)); print("Saved",path,tuple(basis.shape))
