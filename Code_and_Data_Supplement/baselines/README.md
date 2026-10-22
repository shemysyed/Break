# Controlled Baseline Adapters

The paper compares methods under one shared patient-disjoint backbone, preprocessing, disease labels, validation protocol, verifier, and evaluation implementation. These are controlled task adaptations rather than direct reproductions of the original tasks.

`configs/baselines.yaml` and `scripts/13_train_baseline.py` expose the shared implementation interface:

- `standard_retrain`: initialize from the foundation checkpoint and train only on retained patients.
- `retain_ft`: initialize from the Teacher and optimize retained disease utility.
- `neggrad_plus`: combine retained descent with a bounded negative forget gradient.
- `forget_mi`: image-only retained-utility objective plus a forget-side representation-confusion proxy.
- `bad_teaching`: distill retained images from the competent Teacher and forget images from an incompetent/shuffled Teacher target.
- `scrub`: retained Teacher-Student distillation with forget-side divergence.
- `salun`: restrict the forgetting update to parameters selected by forget-gradient saliency.
- `privacy_retrain`: foundation initialization, retain-only disease training, and explicit patient-linkage suppression.

Bad Teaching, SCRUB, and SalUn use the same frozen pairwise verifier trained on Teacher representations. Privacy-Retrain is an empirical privacy reference, not a theoretical lower bound.

The exact loss weights used in the reported runs are stored in `configs/baselines.yaml`. Reviewers should compare these controlled adapters under this protocol rather than interpret them as source-task benchmark reproductions.
