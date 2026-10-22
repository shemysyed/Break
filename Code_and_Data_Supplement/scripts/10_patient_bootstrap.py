from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

parser=argparse.ArgumentParser(); parser.add_argument("--predictions",required=True,help="CSV with patient_id,label,teacher_probability,pulse_probability"); parser.add_argument("--repetitions",type=int,default=1000); parser.add_argument("--seed",type=int,default=2027); args=parser.parse_args()
df=pd.read_csv(args.predictions); patients=df["patient_id"].astype(str).unique(); rng=np.random.default_rng(args.seed); differences=[]
for _ in range(args.repetitions):
    sample=rng.choice(patients,size=len(patients),replace=True); pieces=[]
    for draw,patient in enumerate(sample):
        part=df[df["patient_id"].astype(str)==patient].copy(); part["bootstrap_draw"]=draw; pieces.append(part)
    boot=pd.concat(pieces,ignore_index=True)
    try: differences.append(roc_auc_score(boot["label"],boot["teacher_probability"])-roc_auc_score(boot["label"],boot["pulse_probability"]))
    except ValueError: pass
print({"mean":float(np.mean(differences)),"ci95":[float(np.quantile(differences,.025)),float(np.quantile(differences,.975))],"valid_repetitions":len(differences)})
