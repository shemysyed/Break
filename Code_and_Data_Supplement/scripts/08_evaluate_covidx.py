from __future__ import annotations
import argparse
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument("--config",default="configs/covidx_pulse.yaml")
parser.add_argument("--checkpoint",required=True)
parser.add_argument("--output",default=None)
args=parser.parse_args()
command=[sys.executable,"scripts/07_evaluate_nih.py","--config",args.config,"--checkpoint",args.checkpoint]
if args.output: command += ["--output",args.output]
raise SystemExit(subprocess.call(command))
