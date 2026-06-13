from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from distributed_simulator.cli import build_trainer, config_from_args
from distributed_simulator.distributed import (
    destroy_process_context,
    init_process_context,
    resolve_process_device,
)
from distributed_simulator.logging import LOG_FORMAT
from distributed_simulator.trainers import DecentralizedTrainer, SAMTrainer, SyncTrainer
from distributed_simulator.trainers.base import BaseTrainer


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    percent: float


@dataclass(frozen=True)
class TrainerProfile:
    trainer: str
    rank: int
    world_size: int
    device: str
    virtual_workers: int
    local_workers: int
    batches_per_epoch: int
    parameter_storage_mb: float
    peak_memory_mb: float | None
    cold_batch_ms: float
    phases: dict[str, TimingStats]


def build_parser() -> argparse.ArgumentParser:
    from distributed_simulator.cli import build_parser as build_train_parser

    parser = build_train_parser()
    parser.description = "Profile simulated distributed trainer phases."
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument(
        "--profile-evaluation",
        action="store_true",
        help="Measure one complete evaluation, including BatchNorm calibration.",
    )
    parser.add_argument("--json-output", type=Path, help="Write rank-zero profile as JSON.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.profile_steps < 1:
        raise ValueError("--profile-steps must be positive")


def _synchronize(trainer: BaseTrainer) -> None:
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)


def _measure(trainer: BaseTrainer, function: Callable[[], object]) -> float:
    _synchronize(trainer)
    start = time.perf_counter()
    function()
    _synchronize(trainer)
    return (time.perf_counter() - start) * 1000.0


def _batch(trainer: BaseTrainer, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    return trainer._batch_for_training_step(step)


def _sync_step(trainer: SyncTrainer, step: int) -> None:
    trainer.training_step = step
    trainer._compute_local_gradients(_batch(trainer, step))
    trainer._average_gradients_()
    trainer._apply_optimizer_update(trainer._learning_rate(step))


def _sam_step(trainer: SAMTrainer, step: int) -> None:
    trainer.training_step = step
    trainer._compute_sam_gradients(_batch(trainer, step))
    trainer._average_gradients_()
    trainer._apply_optimizer_update(trainer._learning_rate(step))


def _decentralized_step(trainer: DecentralizedTrainer, step: int) -> None:
    trainer.training_step = step
    lr = trainer._learning_rate(step)
    gamma = trainer._mixing_gamma(step, lr)
    batch = _batch(trainer, step)
    if trainer._use_cuda_mixing_overlap():
        pending = trainer._start_mixing(step, gamma=gamma)
        trainer._compute_local_gradients(batch)
        trainer._finish_mixing(pending)
    else:
        trainer._compute_local_gradients(batch)
        trainer._mix_parameters(step, gamma=gamma)
    trainer._apply_optimizer_update(lr)


def _step_function(trainer: BaseTrainer) -> Callable[[int], None]:
    if isinstance(trainer, SyncTrainer):
        return lambda step: _sync_step(trainer, step)
    if isinstance(trainer, SAMTrainer):
        return lambda step: _sam_step(trainer, step)
    if isinstance(trainer, DecentralizedTrainer):
        return lambda step: _decentralized_step(trainer, step)
    raise TypeError(f"unsupported trainer type: {type(trainer).__name__}")


def _measure_breakdown(trainer: BaseTrainer, step: int) -> dict[str, float]:
    trainer.training_step = step
    result = {"batch": _measure(trainer, lambda: _batch(trainer, step))}
    batch = _batch(trainer, step)
    lr = trainer._learning_rate(step)

    if isinstance(trainer, SyncTrainer):
        result["forward_backward"] = _measure(
            trainer, lambda: trainer._compute_local_gradients(batch)
        )
        result["gradient_average"] = _measure(trainer, trainer._average_gradients_)
    elif isinstance(trainer, SAMTrainer):
        result["sam_forward_backward"] = _measure(
            trainer, lambda: trainer._compute_sam_gradients(batch)
        )
        result["gradient_average"] = _measure(trainer, trainer._average_gradients_)
    elif isinstance(trainer, DecentralizedTrainer):
        gamma = trainer._mixing_gamma(step, lr)
        if trainer._use_cuda_mixing_overlap():
            pending = None

            def start_mixing() -> None:
                nonlocal pending
                pending = trainer._start_mixing(step, gamma=gamma)

            result["mix_start"] = _measure(trainer, start_mixing)
            result["forward_backward"] = _measure(
                trainer, lambda: trainer._compute_local_gradients(batch)
            )
            assert pending is not None
            result["mix_finish"] = _measure(trainer, lambda: trainer._finish_mixing(pending))
        else:
            result["forward_backward"] = _measure(
                trainer, lambda: trainer._compute_local_gradients(batch)
            )
            result["mix"] = _measure(trainer, lambda: trainer._mix_parameters(step, gamma=gamma))
    else:
        raise TypeError(f"unsupported trainer type: {type(trainer).__name__}")

    result["optimizer_storage_sync"] = _measure(
        trainer, lambda: trainer._apply_optimizer_update(lr)
    )
    return result


def _stats(values: list[float], total_ms: float) -> TimingStats:
    return TimingStats(
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        min_ms=min(values),
        max_ms=max(values),
        percent=100.0 * statistics.fmean(values) / total_ms if total_ms else 0.0,
    )


def profile_trainer(
    trainer: BaseTrainer,
    *,
    warmup_steps: int,
    profile_steps: int,
    profile_evaluation: bool,
) -> TrainerProfile:
    cold_batch_ms = _measure(trainer, lambda: _batch(trainer, 0))
    step_function = _step_function(trainer)
    for step in range(warmup_steps):
        step_function(step % trainer.batches_per_epoch)

    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)

    samples: dict[str, list[float]] = {}
    for offset in range(profile_steps):
        step = (warmup_steps + offset) % trainer.batches_per_epoch
        breakdown = _measure_breakdown(trainer, step)
        for name, elapsed in breakdown.items():
            samples.setdefault(name, []).append(elapsed)
        samples.setdefault("end_to_end_step", []).append(
            _measure(trainer, lambda step=step: step_function(step))
        )

    if profile_evaluation:
        samples["evaluation"] = [_measure(trainer, lambda: trainer._evaluate_epoch(1, 0.0, 0.0))]

    total_ms = statistics.fmean(samples["end_to_end_step"])
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(trainer.device) / 1024**2
        if trainer.device.type == "cuda"
        else None
    )
    assert trainer.param_storage is not None
    storage_mb = trainer.param_storage.numel() * trainer.param_storage.element_size() / 1024**2
    return TrainerProfile(
        trainer=trainer.trainer_cfg.name,
        rank=trainer.ctx.rank,
        world_size=trainer.ctx.world_size,
        device=str(trainer.device),
        virtual_workers=trainer.cfg.virtual_workers,
        local_workers=trainer.local_worker_count,
        batches_per_epoch=trainer.batches_per_epoch,
        parameter_storage_mb=storage_mb,
        peak_memory_mb=peak_memory_mb,
        cold_batch_ms=cold_batch_ms,
        phases={name: _stats(values, total_ms) for name, values in samples.items()},
    )


def _print_profile(profile: TrainerProfile) -> None:
    print(
        f"trainer={profile.trainer} rank={profile.rank}/{profile.world_size} "
        f"device={profile.device} virtual_workers={profile.virtual_workers} "
        f"local_workers={profile.local_workers} batches_per_epoch={profile.batches_per_epoch}"
    )
    print(
        f"cold_batch_ms={profile.cold_batch_ms:.3f} "
        f"parameter_storage_mb={profile.parameter_storage_mb:.1f} "
        f"peak_memory_mb={profile.peak_memory_mb if profile.peak_memory_mb is not None else 'n/a'}"
    )
    print(
        f"{'phase':<26} {'mean_ms':>10} {'median_ms':>10} "
        f"{'min_ms':>10} {'max_ms':>10} {'step_%':>8}"
    )
    for name, stats in sorted(profile.phases.items(), key=lambda item: -item[1].mean_ms):
        print(
            f"{name:<26} {stats.mean_ms:>10.3f} {stats.median_ms:>10.3f} "
            f"{stats.min_ms:>10.3f} {stats.max_ms:>10.3f} {stats.percent:>8.1f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper(), format=LOG_FORMAT)
    cfg = config_from_args(args)
    device = resolve_process_device(cfg.device)
    ctx = init_process_context(device)
    try:
        run_cfg = cfg.model_copy(
            update={
                "device": str(device),
                "epochs": max(cfg.epochs, 1),
            }
        )
        trainer = build_trainer(run_cfg, ctx)
        profile = profile_trainer(
            trainer,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            profile_evaluation=args.profile_evaluation,
        )
        if ctx.is_distributed:
            for rank in range(ctx.world_size):
                if ctx.rank == rank:
                    _print_profile(profile)
                    sys.stdout.flush()
                dist.barrier()
        else:
            _print_profile(profile)
        if args.json_output is not None and ctx.rank == 0:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(asdict(profile), indent=2) + "\n")
    finally:
        destroy_process_context()


if __name__ == "__main__":
    main()
