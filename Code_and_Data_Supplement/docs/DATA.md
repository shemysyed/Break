# Data Access and Layout

The exact patient-disjoint split definitions used for the paper are included under `data_splits/`. Medical images are not redistributed.

## NIH ChestX-ray14

Download ChestX-ray14 under its original terms and set `data.image_root` in `configs/nih_pulse.yaml` to the directory containing the image files. The included CSVs use image filenames relative to that directory.

| Split | Images | Patients |
|---|---:|---:|
| retain_train.csv | 78,502 | 21,579 |
| forget_train.csv | 11,274 | 3,098 |
| audit_nonmember_trainlike.csv | 10,926 | 3,033 |
| internal_validation.csv | 5,943 | 1,567 |
| test.csv | 5,475 | 1,528 |

Required modeling columns are `Image Index`, `Patient ID`, and either the 14 disease columns or `Finding Labels`.

## COVIDx CXR

Download COVIDx under its original terms and set `data.image_root` in `configs/covidx_pulse.yaml` to the dataset root containing the `train/`, `val/`, and `test/` directories. Included `image_path` entries are relative paths such as `train/example.png`.

| Split | Images | Patients |
|---|---:|---:|
| retain_train.csv | 47,823 | 24,870 |
| forget_train.csv | 6,703 | 3,529 |
| audit_nonmember_trainlike.csv | 6,604 | 3,529 |
| internal_validation.csv | 6,733 | 3,529 |
| official_validation.csv | 8,473 | 5,163 |
| official_test.csv | 8,482 | 4,722 |

Required columns are `PatientID`, `ImageID`, `label_id`, and `image_path`.

## Validation

From the package root, the exact partitions and zero-overlap property can be checked without images:

```bash
python scripts/01_validate_splits.py --config configs/nih_pulse.yaml --skip-image-check
python scripts/01_validate_splits.py --config configs/covidx_pulse.yaml --skip-image-check
```

Dataset-provided pseudonymous patient identifiers are retained only to define same-patient groups and patient-disjoint partitions. The archive contains no medical images, direct clinical identifiers, author paths, or account information.
