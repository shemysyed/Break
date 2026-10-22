# Expected Outputs

For the selected NIH PULSE checkpoint, the evaluator should report approximately:

- retain macro-AUROC: 0.8595
- forget macro-AUROC: 0.8549
- fixed-direction Pair ReID AUC: 0.4614
- CMC Rank-1: 0.2073
- CMC Rank-5: 0.3030
- mAP@R: 0.0586
- DropR: 0.8226
- separately trained learned pair-audit AUC: 0.6162

Small differences can arise from library versions, pair construction, checkpoint serialization, and floating-point execution. The pair seed, fixed score direction, and evaluation population must match before comparing values.

## Preliminary fixed-K=128

- selected epoch: 9
- alpha: 0.272994608
- beta: 0.224995002
- retain macro-AUROC: 0.904343
- forget macro-AUROC: 0.904539
- Pair ReID AUC: 0.725082
- CMC Rank-1: 0.531414
- CMC Rank-5: 0.640786
- mAP@R: 0.195064
- DropR: 0.409292
