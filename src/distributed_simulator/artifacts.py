from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import tomli_w
import torch
import torch.distributed as dist
from packed_resnet import PackedWideResNet, WideResNet

from distributed_simulator.config import SimulationConfig
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.model import ModelName, get_model
from distributed_simulator.trainers import DecentralizedTrainer, TrainMetrics
from distributed_simulator.trainers.base import BaseTrainer

_STATS_COLUMNS = (
    "epoch",
    "train_loss",
    "test_loss",
    "test_accuracy",
    "distance_to_consensus",
    "lr",
    "gamma",
    "accumulated_gamma",
)


def run_id_from_environment(now: datetime | None = None) -> str:
    slurm_job_id = os.environ.get("SLURM_JOBID")
    if slurm_job_id:
        return slurm_job_id
    current = now or datetime.now()
    return current.strftime("%Y%m%d-%H%M%S")


def create_run_dir(cfg: SimulationConfig, run_id: str | None = None) -> Path:
    path = cfg.logging.root / (run_id or run_id_from_environment())
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_resolved_config(cfg: SimulationConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        tomli_w.dump(cfg.model_dump(mode="json"), file)


def save_stats_csv(metrics: TrainMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_STATS_COLUMNS)
        writer.writeheader()
        for item in metrics.history:
            writer.writerow(
                {
                    "epoch": item.epoch,
                    "train_loss": item.train_loss,
                    "test_loss": item.test_loss,
                    "test_accuracy": item.test_accuracy,
                    "distance_to_consensus": item.distance_to_consensus,
                    "lr": item.lr,
                    "gamma": item.gamma,
                    "accumulated_gamma": item.accumulated_gamma,
                }
            )


def save_last_checkpoints(trainer: BaseTrainer, checkpoint_dir: Path) -> None:
    local_states = _local_worker_state_dicts(trainer)
    gathered = _gather_local_state_dicts(local_states, trainer.ctx)
    if trainer.ctx.rank != 0:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_local_states = [item for rank_items in gathered for item in rank_items]
    all_local_states.sort(key=lambda item: item[0])
    global_state = _average_state_dicts([state for _, state in all_local_states])
    torch.save(global_state, checkpoint_dir / "global_last.pth")

    if isinstance(trainer, DecentralizedTrainer):
        for worker_rank, state in all_local_states:
            torch.save(state, checkpoint_dir / f"local_worker_{worker_rank}.pth")


def _gather_local_state_dicts(
    local_states: list[tuple[int, dict[str, torch.Tensor]]],
    ctx: ProcessContext,
) -> list[list[tuple[int, dict[str, torch.Tensor]]]]:
    if not ctx.is_distributed:
        return [local_states]
    gathered: list[list[tuple[int, dict[str, torch.Tensor]]]] | None = (
        [cast(list[tuple[int, dict[str, torch.Tensor]]], []) for _ in range(ctx.world_size)]
        if ctx.rank == 0
        else None
    )
    dist.gather_object(local_states, object_gather_list=gathered, dst=0)
    return gathered or []


def _local_worker_state_dicts(trainer: BaseTrainer) -> list[tuple[int, dict[str, torch.Tensor]]]:
    assert trainer.model is not None
    if isinstance(trainer.model, PackedWideResNet):
        return _local_wide_resnet_state_dicts(trainer)
    if trainer.cfg.model.name == ModelName.LINEAR:
        return _local_linear_state_dicts(trainer)
    raise TypeError(f"unsupported checkpoint model type: {type(trainer.model).__name__}")


def _local_wide_resnet_state_dicts(
    trainer: BaseTrainer,
) -> list[tuple[int, dict[str, torch.Tensor]]]:
    assert trainer.model is not None
    model = trainer.model
    assert isinstance(model, PackedWideResNet)
    model.sync_storage_from_parameters_()
    states = []
    packed_buffers = dict(model.named_buffers())
    for local_idx, worker_rank in enumerate(trainer.owned_ranks):
        single = WideResNet(
            depth=model.depth,
            widen_factor=model.widen_factor,
            num_classes=model.num_classes,
            in_channels=model.in_channels,
        )
        single.to(device=model.parameter_storage.device, dtype=model.parameter_storage.dtype)
        with torch.no_grad():
            single.parameter_storage.copy_(model.parameter_storage[local_idx : local_idx + 1])
            single.sync_parameters_from_storage_()
            _copy_local_buffers_(single, packed_buffers, local_idx)
        states.append((worker_rank, _state_dict_to_cpu(single.state_dict())))
    return states


def _copy_local_buffers_(
    single: WideResNet,
    packed_buffers: dict[str, torch.Tensor],
    local_idx: int,
) -> None:
    single_buffers = dict(single.named_buffers())
    for name, single_buffer in single_buffers.items():
        if name == "parameter_storage":
            continue
        packed_buffer = packed_buffers[name]
        if single_buffer.ndim == 0:
            single_buffer.copy_(packed_buffer)
            continue
        local_numel = single_buffer.numel()
        start = local_idx * local_numel
        end = start + local_numel
        single_buffer.copy_(packed_buffer.reshape(-1)[start:end].view_as(single_buffer))


def _local_linear_state_dicts(trainer: BaseTrainer) -> list[tuple[int, dict[str, torch.Tensor]]]:
    assert trainer.model is not None
    model = trainer.model
    states = []
    for local_idx, worker_rank in enumerate(trainer.owned_ranks):
        single = get_model(ModelName.LINEAR, trainer.cfg.data.num_classes)
        single.to(device=trainer.device)
        state = single.state_dict()
        state["layers.0.weight"].copy_(cast(Any, model).weight[local_idx])
        state["layers.0.bias"].copy_(cast(Any, model).bias[local_idx])
        states.append((worker_rank, _state_dict_to_cpu(state)))
    return states


def _state_dict_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot average an empty state dict list")
    averaged: dict[str, torch.Tensor] = {}
    keys = states[0].keys()
    for key in keys:
        values = [state[key] for state in states]
        if values[0].is_floating_point():
            averaged[key] = torch.stack(values).mean(dim=0).to(dtype=values[0].dtype)
        else:
            averaged[key] = torch.stack(values).max(dim=0).values
    return averaged
