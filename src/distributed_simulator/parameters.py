from __future__ import annotations

import torch
from torch import nn


def parameters_to_vector(module: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(module.parameters()).detach().clone()


def vector_to_parameters(vector: torch.Tensor, module: nn.Module) -> None:
    torch.nn.utils.vector_to_parameters(vector, module.parameters())


def average_distance_to_consensus(vectors: torch.Tensor) -> float:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [workers, parameters]")
    center = vectors.mean(dim=0, keepdim=True)
    return torch.linalg.vector_norm(vectors - center, dim=1).mean().item()
