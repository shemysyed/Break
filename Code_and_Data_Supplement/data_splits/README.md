# Included patient-disjoint split definitions

This directory contains the exact image-level partitions used in the reported experiments.
No medical images are redistributed.

- `nih/`: NIH ChestX-ray14 retain, forget, audit, internal-validation, and test partitions.
- `covidx/`: COVIDx retain, forget, audit, internal-validation, official-validation, and official-test partitions.

Local author filesystem paths were removed. In the CSV files, `ImagePath`/`image_path` values are relative to the dataset root configured in the YAML files. Dataset-provided pseudonymous patient identifiers are retained because they define same-patient grouping and the patient-disjoint protocol; they are not author identifiers.
