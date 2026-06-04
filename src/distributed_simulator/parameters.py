from __future__ import annotations

import torch


def average_distance_to_consensus(vectors: torch.Tensor) -> float:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [workers, parameters]")
    center = vectors.mean(dim=0, keepdim=True)
    return torch.linalg.vector_norm(vectors - center, dim=1).mean().item()
