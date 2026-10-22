from __future__ import annotations
import argparse
from pathlib import Path
from pulse.config import load_config
from pulse.data import load_split, validate_patient_disjoint

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--skip-image-check", action="store_true")
args = parser.parse_args()
config = load_config(args.config)
data = config["data"]
splits = {}
for name, filename in data["split_files"].items():
    path = Path(data["split_dir"]) / filename
    if not path.exists():
        print(f"SKIP {name}: {path} is absent")
        continue
    splits[name] = load_split(
        path, data["image_root"], data["labels"], data["image_column"], data["patient_column"],
        data.get("path_column", "ImagePath"), data.get("label_column"), not args.skip_image_check,
    )
    expected = data.get("expected_counts", {}).get(name)
    actual = {"images": len(splits[name]), "patients": splits[name][data["patient_column"]].nunique()}
    print(name, actual)
    if expected and actual != expected:
        raise ValueError(f"Count mismatch for {name}: expected {expected}, observed {actual}")
validate_patient_disjoint(splits, data["patient_column"])
print("Patient-disjoint validation passed.")
