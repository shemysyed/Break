# PULSE - AAAI-27 Code and Data Supplement

Anonymous code supplement for:

**PULSE: Patient-Level Unlearning via Localized Selective Erasure for Privacy-Preserving Chest X-ray Foundation Models**

This package provides a clean, command-line implementation of the method, data protocol, training objectives, evaluation metrics, and configurations described in the paper. It is organized from the authors' experimental notebooks, with exploratory outputs, personal filesystem paths, and failed runs removed.

## What is included

- patient-disjoint split validation for NIH ChestX-ray14 and COVIDx CXR;
- Teacher and retain-only retraining entry points;
- disease-matched retain-anchor construction;
- identity-sensitive subspace extraction;
- six-effect disease-protected interventional token scoring;
- PULSE Student and learned pairwise attacker;
- final NIH training configuration and deterministic clean-image inference;
- fixed-direction Pair ReID, CMC Rank-1/5, mAP@R, and DropR evaluation;
- patient-level bootstrap utilities;
- fixed-cardinality analysis utilities;
- controlled baseline-adaptation entry point and learned-linkage audit;
- figure reproduction scripts;
- exact anonymized patient-disjoint split definitions, reported numerical results, and expected split statistics;
- pinned software environment and a reproducibility checklist map.

## What is not redistributed

NIH ChestX-ray14 and COVIDx medical images, together with the `microsoft/rad-dino-maira-2` foundation weights, are governed by their original access and licensing terms and are not copied into this archive. The exact split CSVs are included with local paths replaced by relative paths. Large trained checkpoints are also not embedded in the ZIP. `docs/DATA.md` and `checkpoints/README.md` specify the expected locations, schemas, and filenames.

## Reproduction levels

### 1. Verify package and reported tables

```bash
python -m pip install -e .
python scripts/00_verify_environment.py
python scripts/01_validate_splits.py --config configs/nih_pulse.yaml --skip-image-check
python scripts/01_validate_splits.py --config configs/covidx_pulse.yaml --skip-image-check
python scripts/11_show_reported_results.py
pytest -q
```

### 2. Evaluate supplied checkpoints

Place the checkpoints described in `checkpoints/README.md`, then run:

```bash
python scripts/07_evaluate_nih.py \
  --config configs/nih_pulse.yaml \
  --checkpoint checkpoints/pulse_nih_selected.pt
```

### 3. Reproduce from the foundation checkpoint

```bash
python scripts/02_train_teacher.py --config configs/nih_teacher.yaml
python scripts/03_cache_anchors.py --config configs/nih_pulse.yaml
python scripts/04_cache_identity_subspace.py --config configs/nih_pulse.yaml
python scripts/05_score_tokens.py --config configs/nih_pulse.yaml
python scripts/06_train_pulse.py --config configs/nih_pulse.yaml
python scripts/07_evaluate_nih.py --config configs/nih_pulse.yaml \
  --checkpoint outputs/nih/pulse_selected.pt
```

The reported final refinement uses one NVIDIA A100-SXM4-80GB GPU. Offline token scoring is the most expensive preprocessing stage and is cached before Student optimization.

## Core NIH configuration

- backbone: `microsoft/rad-dino-maira-2`
- input: 224 x 224
- disease outputs: 14
- projected embedding: 256
- privacy bottleneck: 32
- identity-sensitive subspace: 96
- training token cardinalities: 77, 96, 128
- sampling probabilities: 0.20, 0.35, 0.45
- trainable encoder blocks: final two
- Student updates: 2,500 (5 x 500)
- Student learning rate: 1e-6
- attacker learning rate: 1.5e-4
- final inference point: alpha=0.70, beta=0.631997
- embedding noise: none

## Important evaluation convention

Pair ReID AUC uses raw cosine similarity with the fixed direction **higher similarity = same patient**. Scores are never inverted. Thus, values below 0.5 indicate reversed score ordering rather than being reflected above 0.5.

## Package map

See `docs/FILE_MAP.md`. The AAAI reproducibility mapping is in `docs/REPRODUCIBILITY_CHECKLIST.md`.

## Reproducibility status

The package code and configuration are aligned to the method described in the submitted paper. The included smoke tests validate deterministic utilities and metric conventions without downloading medical data. Exact end-to-end numerical reproduction requires the licensed datasets, the foundation checkpoint, and the trained checkpoint files listed in `checkpoints/README.md`.
