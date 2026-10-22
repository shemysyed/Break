# AAAI-27 Reproducibility Checklist Map

The official AAAI-27 checklist must be completed and uploaded separately. This document maps reproducibility information to files in this Code and Data Supplement.

## Problem, claims, and assumptions

- Research question and threat model: main paper and supplementary document.
- Pair ReID direction and interpretation: `README.md`, `src/pulse/metrics.py`.
- Scope limits: `README.md` and `docs/DATA.md`.

## Method specification

- Model architecture: `src/pulse/model.py`.
- Six-effect token score: `src/pulse/token_scoring.py`.
- Identity-sensitive subspace: `src/pulse/identity.py`.
- Learned attacker: `src/pulse/attacker.py`.
- Loss definitions: `src/pulse/losses.py` and `scripts/06_train_pulse.py`.
- Full hyperparameters: `configs/nih_pulse.yaml`.

## Data and splits

- Dataset access and schema: `docs/DATA.md`.
- Exact NIH split counts: `configs/nih_pulse.yaml`.
- Patient-overlap check: `scripts/01_validate_splits.py`.
- The datasets are not redistributed because of original terms and medical-data considerations.

## Training and model selection

- Teacher training: `scripts/02_train_teacher.py`.
- Offline cached assets: scripts 03-05.
- PULSE training: `scripts/06_train_pulse.py`.
- Internal validation is used for checkpoint and operating-point selection.
- Audit and held-out test populations are excluded from training and model selection.
- Each reported model configuration was trained once; evaluation uncertainty is estimated by patient-level bootstrap rather than independent training seeds.

## Randomness

- Teacher seed: 42.
- Final refinement seed: 808.
- Evaluation seed: 1212.
- Pair construction seed: 1229.
- Bootstrap seed: 2027.
- Seed utilities: `src/pulse/utils.py`.

## Evaluation

- Disease macro-AUROC: `src/pulse/metrics.py`.
- Pair ReID with 50,000 balanced pairs: `src/pulse/metrics.py`.
- CMC Rank-1/5 and mAP@R: `src/pulse/metrics.py`.
- DropR definition: `src/pulse/metrics.py`.
- Patient-level bootstrap: `scripts/10_patient_bootstrap.py`.

## Compute and software

- OS: Ubuntu 22.04.5 LTS.
- Python: 3.12.13.
- PyTorch: 2.11.0+cu128.
- CUDA: 12.8.
- cuDNN: 9.19.0.
- Torchvision: 0.26.0+cu128.
- Transformers: 5.13.1.
- scikit-learn: 1.6.1.
- NumPy: 2.0.2.
- Pandas: 2.2.2.
- Pillow: 11.3.0.
- GPU: NVIDIA A100-SXM4-80GB.
- Verification command: `scripts/00_verify_environment.py`.

## Computational cost

- Offline token scoring: O(N_f M) frozen-Teacher interventions.
- NIH values: N_f=11,274 and M=256, giving 2,886,144 interventions.
- Final PULSE trainable parameters: 14,396,462.
- Final refinement: 2,500 Student updates.
- 500 refinement updates: approximately 2 min 53 s to 2 min 55 s on one A100-SXM4-80GB.
- Offline token-scoring time is not claimed because it was not separately logged.

## Results and expected outputs

- Main NIH table: `results/nih_main_results.csv`.
- NIH operating points: `results/nih_operating_points.csv`.
- COVIDx table: `results/covidx_results.csv`.
- Fixed-cardinality analysis: `results/fixed_k_full_metrics.csv`.

## Artifact limitations

Exact numerical reproduction requires licensed datasets, the foundation checkpoint, and the trained checkpoint binaries. The package does not claim that its smoke test reproduces the full reported checkpoint; the smoke test verifies code paths and metric conventions only.
