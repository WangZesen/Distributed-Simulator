from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ProcessContext:
    rank: int = 0
    world_size: int = 1

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def init_process_context(device: torch.device) -> ProcessContext:
    if dist.is_available() and dist.is_initialized():
        return ProcessContext(rank=dist.get_rank(), world_size=dist.get_world_size())

    if not dist.is_available():
        return ProcessContext()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return ProcessContext()

    backend = "gloo" if device.type == "cpu" else "nccl"
    dist.init_process_group(backend=backend, init_method="env://")
    return ProcessContext(rank=dist.get_rank(), world_size=dist.get_world_size())


def destroy_process_context() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def resolve_process_device(requested_device: str | torch.device) -> torch.device:
    device = torch.device(requested_device)
    if device.type != "cuda":
        return device

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and device.index is None:
        device = torch.device(f"cuda:{int(local_rank)}")

    if torch.cuda.is_available() and device.index is not None:
        torch.cuda.set_device(device)
    return device


def all_gather_owned_buckets(local_buckets: torch.Tensor, ctx: ProcessContext) -> torch.Tensor:
    if not ctx.is_distributed:
        return local_buckets

    gathered = [torch.empty_like(local_buckets) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local_buckets.contiguous())
    return torch.cat(gathered, dim=0)
