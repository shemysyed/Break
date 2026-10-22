# Checkpoints

Large checkpoint binaries are not embedded in this lightweight archive. Place them under `checkpoints/` with these anonymous names:

| File | Purpose |
|---|---|
| `teacher_nih.pt` | Frozen NIH Teacher with disease and patient heads |
| `standard_retrain_nih.pt` | Retain-only Standard Retrain |
| `pulse_nih_selected.pt` | Selected final NIH PULSE checkpoint (epoch 4) |
| `teacher_covidx.pt` | COVIDx Teacher |
| `pulse_covidx_selected.pt` | Selected COVIDx PULSE checkpoint |
| `fixed_k/k64.pt` | Preliminary fixed-K=64 checkpoint |
| `fixed_k/k77.pt` | Preliminary fixed-K=77 checkpoint |
| `fixed_k/k96.pt` | Preliminary fixed-K=96 checkpoint |
| `fixed_k/k128.pt` | Preliminary fixed-K=128 checkpoint (selected epoch 9) |

The final NIH checkpoint should contain at least:

- `model_state_dict`
- `epoch`
- configuration metadata
- a 96-dimensional `identity_basis`
- `default_alpha` and `default_beta` buffers

The evaluation script also accepts a raw state dictionary.
