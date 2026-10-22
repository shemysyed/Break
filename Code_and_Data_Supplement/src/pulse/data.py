from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _basename(value: object) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def load_split(
    csv_path: str | Path,
    image_root: str | Path,
    labels: list[str],
    image_column: str = "Image Index",
    patient_column: str = "Patient ID",
    path_column: str = "ImagePath",
    label_column: str | None = None,
    verify_images: bool = True,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    image_root = Path(image_root)
    df = pd.read_csv(csv_path)
    for required in (image_column, patient_column):
        if required not in df.columns:
            raise KeyError(f"{required!r} is missing from {csv_path}")
    df[image_column] = df[image_column].astype(str)
    df[patient_column] = df[patient_column].astype(str)
    # Use anonymized relative paths when supplied; otherwise rebuild from the image identifier.
    if path_column in df.columns:
        raw_paths = df[path_column].fillna("").astype(str).str.replace("\\", "/", regex=False)
        is_relative = ~raw_paths.str.match(r"^(?:/|[A-Za-z]:/)") & raw_paths.ne("")
        rebuilt = df[image_column].map(lambda value: str(image_root / _basename(value)))
        relative = raw_paths.map(lambda value: str(image_root / Path(value)))
        df[path_column] = np.where(is_relative, relative, rebuilt)
    else:
        df[path_column] = df[image_column].map(lambda value: str(image_root / _basename(value)))
    if label_column is None:
        for label in labels:
            if label not in df.columns:
                if "Finding Labels" not in df.columns:
                    raise KeyError(f"Missing label {label!r} in {csv_path}")
                df[label] = df["Finding Labels"].map(
                    lambda text, target=label: int(target in str(text).split("|"))
                )
            df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(np.float32)
    else:
        if label_column not in df.columns:
            raise KeyError(f"Missing label column {label_column!r} in {csv_path}")
        df[labels[0]] = pd.to_numeric(df[label_column], errors="raise").astype(np.float32)
    if verify_images:
        missing = ~df[path_column].map(lambda value: Path(value).is_file())
        if missing.any():
            examples = df.loc[missing, [image_column, path_column]].head(5)
            raise FileNotFoundError(
                f"{int(missing.sum()):,} images are missing for {csv_path}.\n{examples.to_string(index=False)}"
            )
    return df.reset_index(drop=True)


def validate_patient_disjoint(splits: dict[str, pd.DataFrame], patient_column: str) -> None:
    names = list(splits)
    for i, left in enumerate(names):
        left_ids = set(splits[left][patient_column].astype(str))
        for right in names[i + 1:]:
            overlap = left_ids.intersection(set(splits[right][patient_column].astype(str)))
            if overlap:
                raise ValueError(f"Patient overlap between {left} and {right}: {len(overlap)}")


class ImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform,
        labels: list[str],
        image_column: str,
        patient_column: str,
        path_column: str,
        masks: np.ndarray | None = None,
        anchors: np.ndarray | None = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.labels = self.df[labels].to_numpy(dtype=np.float32)
        self.image_column = image_column
        self.patient_column = patient_column
        self.path_column = path_column
        self.masks = masks
        self.anchors = anchors
        if masks is not None and len(masks) != len(self.df):
            raise ValueError("Mask count does not match dataframe length")
        if anchors is not None and len(anchors) != len(self.df):
            raise ValueError("Anchor count does not match dataframe length")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        with Image.open(row[self.path_column]) as image:
            pixel_values = self.transform(image.convert("RGB"))
        item: dict[str, Any] = {
            "pixel_values": pixel_values,
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
            "patient_id": str(row[self.patient_column]),
            "image_id": str(row[self.image_column]),
            "row_index": index,
        }
        if self.masks is not None:
            item["mask_indices"] = torch.tensor(self.masks[index], dtype=torch.long)
        if self.anchors is not None:
            item["anchor"] = torch.tensor(self.anchors[index], dtype=torch.float32)
        return item


class SamePatientPairDataset(Dataset):
    def __init__(self, base: ImageDataset, length: int = 200_000, seed: int = 808) -> None:
        self.base = base
        self.length = int(length)
        self.rng = np.random.default_rng(seed)
        groups: dict[str, list[int]] = defaultdict(list)
        for index, patient_id in enumerate(base.df[base.patient_column].astype(str)):
            groups[patient_id].append(index)
        self.groups = {key: value for key, value in groups.items() if len(value) >= 2}
        self.patient_ids = sorted(self.groups)
        if not self.patient_ids:
            raise ValueError("No patient has at least two images")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _: int) -> dict[str, Any]:
        patient_id = self.patient_ids[int(self.rng.integers(0, len(self.patient_ids)))]
        first, second = self.rng.choice(self.groups[patient_id], size=2, replace=False)
        return {"first": self.base[int(first)], "second": self.base[int(second)]}
