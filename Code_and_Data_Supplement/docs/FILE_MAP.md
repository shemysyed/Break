# File Map

| Path | Purpose |
|---|---|
| `data_splits/` | Exact anonymized NIH and COVIDx patient-disjoint partitions |
| `configs/nih_pulse.yaml` | Final NIH PULSE hyperparameters and paths |
| `configs/nih_teacher.yaml` | Teacher training configuration |
| `configs/covidx_pulse.yaml` | COVIDx protocol configuration |
| `src/pulse/model.py` | Teacher and PULSE Student architectures |
| `src/pulse/token_scoring.py` | Six-effect disease-protected token scoring |
| `src/pulse/identity.py` | Identity-sensitive subspace extraction |
| `src/pulse/attacker.py` | Learned pairwise linkage attacker |
| `src/pulse/losses.py` | Utility and privacy loss primitives |
| `src/pulse/metrics.py` | AUROC, fixed-direction Pair ReID, CMC, mAP@R, DropR |
| `scripts/02_train_teacher.py` | Teacher training |
| `scripts/03_cache_anchors.py` | Disease-matched retain anchors |
| `scripts/04_cache_identity_subspace.py` | SVD identity basis |
| `scripts/05_score_tokens.py` | Cached six-effect token rankings |
| `scripts/06_train_pulse.py` | Final PULSE refinement |
| `scripts/07_evaluate_nih.py` | Full NIH evaluation |
| `scripts/10_patient_bootstrap.py` | Patient-level bootstrap |
| `results/` | Reported paper and supplementary values |
| `docs/REPRODUCIBILITY_CHECKLIST.md` | Checklist-to-artifact mapping |

| `scripts/13_train_baseline.py` | Shared controlled baseline adaptations |
| `scripts/14_learned_linkage_audit.py` | Held-out learned pair audit |
| `scripts/figures/plot_multimetric.py` | Main multi-metric figure |
