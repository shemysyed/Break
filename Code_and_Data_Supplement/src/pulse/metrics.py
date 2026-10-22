from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_auroc(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[float]]:
    values: list[float] = []
    for column in range(labels.shape[1]):
        if np.unique(labels[:, column]).size < 2:
            values.append(float("nan"))
        else:
            values.append(float(roc_auc_score(labels[:, column], probabilities[:, column])))
    return float(np.nanmean(values)), values


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)


def pair_reid_auc(embeddings: np.ndarray, patient_ids: np.ndarray, n_pairs: int = 50_000,
                    seed: int = 1229) -> float:
    """Fixed direction: high raw cosine similarity predicts same patient. No inversion."""
    embeddings = normalize_embeddings(embeddings)
    patient_ids = np.asarray(patient_ids).astype(str)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(patient_ids):
        groups[patient].append(index)
    repeated = [patient for patient, indices in groups.items() if len(indices) >= 2]
    patients = np.asarray(list(groups), dtype=object)
    if not repeated:
        raise ValueError("No patient has at least two images")
    rng = np.random.default_rng(seed)
    positive_count = n_pairs // 2
    scores = np.empty(n_pairs, dtype=np.float32)
    labels = np.empty(n_pairs, dtype=np.int8)
    for row in range(positive_count):
        patient = repeated[int(rng.integers(0, len(repeated)))]
        first, second = rng.choice(groups[patient], size=2, replace=False)
        scores[row] = embeddings[first] @ embeddings[second]
        labels[row] = 1
    for row in range(positive_count, n_pairs):
        first_patient, second_patient = rng.choice(patients, size=2, replace=False)
        first = rng.choice(groups[first_patient])
        second = rng.choice(groups[second_patient])
        scores[row] = embeddings[first] @ embeddings[second]
        labels[row] = 0
    return float(roc_auc_score(labels, scores))


def retrieval_metrics(embeddings: np.ndarray, patient_ids: np.ndarray, chunk_size: int = 256) -> dict[str, float]:
    """All-other-images gallery. mAP@R uses R = number of relevant gallery images."""
    embeddings = normalize_embeddings(embeddings)
    patient_ids = np.asarray(patient_ids).astype(str)
    counts = pd.Series(patient_ids).value_counts().to_dict()
    queries = np.asarray([i for i, patient in enumerate(patient_ids) if counts[patient] >= 2], dtype=np.int64)
    rank1 = rank5 = 0
    ap_values: list[float] = []
    for start in range(0, len(queries), chunk_size):
        query_indices = queries[start:start + chunk_size]
        similarities = embeddings[query_indices] @ embeddings.T
        similarities[np.arange(len(query_indices)), query_indices] = -np.inf
        orders = np.argsort(-similarities, axis=1)
        for row, query in enumerate(query_indices):
            patient = patient_ids[query]
            relevant_count = counts[patient] - 1
            relevance = (patient_ids[orders[row]] == patient).astype(np.int32)
            rank1 += int(relevance[0])
            rank5 += int(relevance[:5].any())
            top_r = relevance[:relevant_count]
            precision = np.cumsum(top_r) / (np.arange(relevant_count, dtype=np.float32) + 1.0)
            ap_values.append(float((precision * top_r).sum() / relevant_count))
    count = len(ap_values)
    if count == 0:
        raise ValueError("No valid retrieval queries")
    return {"queries": count, "cmc_rank1": rank1 / count, "cmc_rank5": rank5 / count,
            "map_at_r": float(np.mean(ap_values))}


def drop_r(method_map_at_r: float, teacher_map_at_r: float) -> float:
    return float(1.0 - method_map_at_r / teacher_map_at_r)
