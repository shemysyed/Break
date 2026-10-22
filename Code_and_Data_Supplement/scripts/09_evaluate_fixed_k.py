from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from pulse.config import load_config

parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/fixed_k_v3.yaml"); parser.add_argument("--results",default="results/fixed_k_full_metrics.csv"); args=parser.parse_args()
cfg=load_config(args.config); results=pd.read_csv(args.results); expected={int(row["k"]) for row in cfg["models"]}; present=set(results["K"].astype(int)); print(results.to_string(index=False)); missing=expected-present
if missing: print("Pending fixed-K evaluations:",sorted(missing))
else: print("All configured fixed-K rows are present.")
