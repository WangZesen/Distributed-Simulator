from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@dataclass
class HvpBatchSize:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("HVP batch size must be positive")

    def halve(self) -> int:
        if self.value <= 1:
            raise torch.OutOfMemoryError("cannot reduce HVP batch size below 1")
        self.value = max(1, self.value // 2)
        return self.value


@dataclass(frozen=True)
class LanczosEigenvalueEstimate:
    eigenvalues: tuple[float, ...]
    lambda_min: float
    lambda_max: float


def parameter_dot(v1: Sequence[torch.Tensor], v2: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(v1) != len(v2):
        raise ValueError("parameter vector lengths must match")
    if not v1:
        raise ValueError("parameter vectors must not be empty")
    total = torch.zeros((), device=v1[0].device, dtype=v1[0].dtype)
    for a, b in zip(v1, v2, strict=True):
        total = total + torch.sum(a * b)
    return total


def normalize_parameters(
    vector: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    norm = torch.sqrt(parameter_dot(vector, vector))
    if norm.item() == 0.0:
        raise ValueError("cannot normalize a zero parameter vector")
    return [item / norm for item in vector], norm


def minibatch_hvp(
    model: nn.Module,
    parameters: Sequence[nn.Parameter],
    batch: tuple[torch.Tensor, torch.Tensor],
    vector: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    inputs, targets = batch
    with torch.enable_grad():
        loss = F.cross_entropy(model(inputs), targets, reduction="sum")
        grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=False)
        hvp = torch.autograd.grad(
            grads,
            parameters,
            grad_outputs=vector,
            retain_graph=False,
            allow_unused=False,
        )
    return list(hvp)


def full_dataset_hvp(
    model: nn.Module,
    full_batch: tuple[torch.Tensor, torch.Tensor],
    batch_size: HvpBatchSize,
    vector: Sequence[torch.Tensor],
    *,
    drop_last: bool = False,
) -> list[torch.Tensor]:
    return _full_dataset_hvp_with_oom_fallback(
        model,
        full_batch,
        batch_size,
        vector,
        drop_last=drop_last,
    )


def _full_dataset_hvp_once(
    model: nn.Module,
    parameters: Sequence[nn.Parameter],
    full_batch: tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    vector: Sequence[torch.Tensor],
    *,
    drop_last: bool,
) -> list[torch.Tensor]:
    if len(parameters) != len(vector):
        raise ValueError("HVP vector length must match model parameter count")

    carry = [torch.zeros_like(parameter) for parameter in parameters]
    inputs, targets = full_batch
    num_samples = inputs.size(0)
    processed = 0
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        if drop_last and end - start < batch_size:
            break
        hvp = minibatch_hvp(model, parameters, (inputs[start:end], targets[start:end]), vector)
        torch._foreach_add_(carry, hvp)
        processed += end - start

    if processed == 0:
        raise ValueError(
            f"no HVP batches available for batch size {batch_size} and dataset size {num_samples}"
        )

    count = torch.tensor(float(processed), device=carry[0].device, dtype=carry[0].dtype)
    if dist.is_available() and dist.is_initialized():
        for item in carry:
            dist.all_reduce(item, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    torch._foreach_div_(carry, count)
    return carry


def _full_dataset_hvp_with_oom_fallback(
    model: nn.Module,
    full_batch: tuple[torch.Tensor, torch.Tensor],
    batch_size: HvpBatchSize,
    vector: Sequence[torch.Tensor],
    *,
    drop_last: bool,
) -> list[torch.Tensor]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    while True:
        try:
            return _full_dataset_hvp_once(
                model,
                parameters,
                full_batch,
                batch_size.value,
                vector,
                drop_last=drop_last,
            )
        except torch.OutOfMemoryError:
            if batch_size.value <= 1:
                raise
            gc.collect()
            first_device = parameters[0].device if parameters else torch.device("cpu")
            if first_device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size.halve()
            logger.warning("OOM during HVP; retrying with batch size {}", batch_size.value)


@torch.no_grad()
def lanczos_eigenvalues(
    model: nn.Module,
    full_batch: tuple[torch.Tensor, torch.Tensor],
    batch_size: HvpBatchSize,
    *,
    num_iters: int,
    seed: int,
    tolerance: float = 1e-6,
) -> LanczosEigenvalueEstimate:
    if num_iters < 1:
        raise ValueError("num_iters must be positive")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")

    device = parameters[0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    q, _ = normalize_parameters(
        [
            torch.randn(parameter.shape, generator=generator, device=device)
            for parameter in parameters
        ]
    )

    if dist.is_available() and dist.is_initialized():
        for item in q:
            dist.broadcast(item, src=0)

    alphas: list[float] = []
    betas: list[float] = []
    beta = 0.0
    q_prev: list[torch.Tensor] | None = None

    for iteration in range(num_iters):
        hq = full_dataset_hvp(model, full_batch, batch_size, q, drop_last=True)
        alpha = parameter_dot(q, hq)
        alphas.append(alpha.item())

        residual = [hqi - alpha * qi for hqi, qi in zip(hq, q, strict=True)]
        if q_prev is not None:
            residual = [
                residual_item - beta * q_prev_item
                for residual_item, q_prev_item in zip(residual, q_prev, strict=True)
            ]

        beta = torch.sqrt(parameter_dot(residual, residual)).item()
        betas.append(beta)
        logger.info(
            "Lanczos iteration {}: alpha={:.6f} beta={:.6f}",
            iteration,
            alpha.item(),
            beta,
        )

        if beta < tolerance:
            break

        q_prev = q
        q = [item / beta for item in residual]

    values = _tridiagonal_eigenvalues(alphas, betas, device=device)
    return LanczosEigenvalueEstimate(
        eigenvalues=tuple(values),
        lambda_min=min(values),
        lambda_max=max(values),
    )


def _tridiagonal_eigenvalues(
    alphas: Sequence[float],
    betas: Sequence[float],
    *,
    device: torch.device,
) -> list[float]:
    size = len(alphas)
    matrix = torch.zeros(size, size, device=device)
    for idx, alpha in enumerate(alphas):
        matrix[idx, idx] = alpha
        if idx + 1 < size:
            matrix[idx, idx + 1] = betas[idx]
            matrix[idx + 1, idx] = betas[idx]
    return torch.linalg.eigvalsh(matrix).cpu().tolist()
