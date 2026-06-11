from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from distributed_simulator.cli import build_trainer, config_from_args
from distributed_simulator.config import SimulationConfig
from distributed_simulator.data import num_classes_for_dataset
from distributed_simulator.distributed import (
    ProcessContext,
    destroy_process_context,
    init_process_context,
    resolve_process_device,
)
from distributed_simulator.model import ModelName, get_packed_model
from distributed_simulator.precision import configure_tf32
from distributed_simulator.trainers.base import BaseTrainer

_IMAGE_SHAPE = (3, 32, 32)


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class ForwardBackwardCase:
    phases: dict[str, TimingStats]
    peak_memory_mb: float | None


@dataclass(frozen=True)
class ForwardBackwardProfile:
    rank: int
    world_size: int
    device: str
    model: str
    trainer: str
    local_workers: int
    local_batch_size: int
    input_shape: tuple[int, ...]
    input_stride: tuple[int, ...]
    input_dtype: str
    channels_last: bool
    amp: bool
    compile: bool
    compile_mode: str
    cudnn_benchmark: bool
    cases: dict[str, ForwardBackwardCase]


def build_parser() -> argparse.ArgumentParser:
    from distributed_simulator.cli import build_parser as build_train_parser

    parser = build_train_parser()
    parser.description = "Compare isolated Packed-ResNet and trainer forward/backward timing."
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cuDNN benchmarking to match benchmark_gpu_timing.py.",
    )
    parser.add_argument("--json-output", type=Path, help="Write rank-zero profile as JSON.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.profile_steps < 1:
        raise ValueError("--profile-steps must be positive")


def _stats(values: list[float]) -> TimingStats:
    return TimingStats(
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        min_ms=min(values),
        max_ms=max(values),
    )


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _packed_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size, num_models, num_classes = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * num_models, num_classes),
        targets.reshape(-1),
    )


def _run_forward_backward(
    model: nn.Module,
    forward_model: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    amp_enabled: bool,
) -> None:
    model.zero_grad(set_to_none=True)
    with _autocast_context(inputs.device, amp_enabled):
        loss = _packed_loss(forward_model(inputs), targets)
    loss.backward()


def _profile_cuda_phases(
    model: nn.Module,
    forward_model: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    amp_enabled: bool,
    profile_steps: int,
) -> dict[str, TimingStats]:
    samples = {name: [] for name in ("forward", "loss", "backward", "total")}
    for _ in range(profile_steps):
        model.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        after_forward = torch.cuda.Event(enable_timing=True)
        after_loss = torch.cuda.Event(enable_timing=True)
        after_backward = torch.cuda.Event(enable_timing=True)
        start.record()
        with _autocast_context(inputs.device, amp_enabled):
            logits = forward_model(inputs)
            after_forward.record()
            loss = _packed_loss(logits, targets)
            after_loss.record()
        loss.backward()
        after_backward.record()
        torch.cuda.synchronize(inputs.device)
        samples["forward"].append(start.elapsed_time(after_forward))
        samples["loss"].append(after_forward.elapsed_time(after_loss))
        samples["backward"].append(after_loss.elapsed_time(after_backward))
        samples["total"].append(start.elapsed_time(after_backward))
    return {name: _stats(values) for name, values in samples.items()}


def _profile_cpu_phases(
    model: nn.Module,
    forward_model: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    amp_enabled: bool,
    profile_steps: int,
) -> dict[str, TimingStats]:
    samples = {name: [] for name in ("forward", "loss", "backward", "total")}
    for _ in range(profile_steps):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with _autocast_context(inputs.device, amp_enabled):
            logits = forward_model(inputs)
            after_forward = time.perf_counter()
            loss = _packed_loss(logits, targets)
            after_loss = time.perf_counter()
        loss.backward()
        after_backward = time.perf_counter()
        samples["forward"].append((after_forward - start) * 1000.0)
        samples["loss"].append((after_loss - after_forward) * 1000.0)
        samples["backward"].append((after_backward - after_loss) * 1000.0)
        samples["total"].append((after_backward - start) * 1000.0)
    return {name: _stats(values) for name, values in samples.items()}


def _profile_phases(
    model: nn.Module,
    forward_model: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    amp_enabled: bool,
    warmup_steps: int,
    profile_steps: int,
) -> ForwardBackwardCase:
    model.train()
    for _ in range(warmup_steps):
        _run_forward_backward(
            model,
            forward_model,
            inputs,
            targets,
            amp_enabled=amp_enabled,
        )
    if inputs.device.type == "cuda":
        torch.cuda.synchronize(inputs.device)
        torch.cuda.reset_peak_memory_stats(inputs.device)
        phases = _profile_cuda_phases(
            model,
            forward_model,
            inputs,
            targets,
            amp_enabled=amp_enabled,
            profile_steps=profile_steps,
        )
        peak_memory_mb = torch.cuda.max_memory_allocated(inputs.device) / 1024**2
    else:
        phases = _profile_cpu_phases(
            model,
            forward_model,
            inputs,
            targets,
            amp_enabled=amp_enabled,
            profile_steps=profile_steps,
        )
        peak_memory_mb = None
    return ForwardBackwardCase(phases=phases, peak_memory_mb=peak_memory_mb)


def _profile_trainer_method(
    trainer: BaseTrainer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    warmup_steps: int,
    profile_steps: int,
) -> ForwardBackwardCase:
    batch = (inputs, targets)
    for _ in range(warmup_steps):
        trainer._compute_local_gradients(batch)
    samples = []
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
        torch.cuda.reset_peak_memory_stats(trainer.device)
        for _ in range(profile_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            trainer._compute_local_gradients(batch)
            end.record()
            torch.cuda.synchronize(trainer.device)
            samples.append(start.elapsed_time(end))
        peak_memory_mb = torch.cuda.max_memory_allocated(trainer.device) / 1024**2
    else:
        for _ in range(profile_steps):
            start = time.perf_counter()
            trainer._compute_local_gradients(batch)
            samples.append((time.perf_counter() - start) * 1000.0)
        peak_memory_mb = None
    return ForwardBackwardCase(phases={"total": _stats(samples)}, peak_memory_mb=peak_memory_mb)


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _random_packed_batch(
    cfg: SimulationConfig,
    *,
    device: torch.device,
    local_workers: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels, height, width = _IMAGE_SHAPE
    torch.manual_seed(cfg.seed)
    if cfg.model.name == ModelName.LINEAR:
        inputs = torch.randn(
            cfg.data.batch_size,
            local_workers,
            channels,
            height,
            width,
            device=device,
        )
    else:
        inputs = torch.randn(
            cfg.data.batch_size,
            local_workers * channels,
            height,
            width,
            device=device,
        ).contiguous(memory_format=torch.channels_last)
    targets = torch.randint(
        num_classes,
        (cfg.data.batch_size, local_workers),
        device=device,
    )
    return inputs, targets


def profile_forward_backward(
    cfg: SimulationConfig,
    ctx: ProcessContext,
    *,
    warmup_steps: int,
    profile_steps: int,
    cudnn_benchmark: bool,
) -> ForwardBackwardProfile:
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(update={"cudnn_benchmark": cudnn_benchmark}),
        }
    )
    device = torch.device(cfg.device)
    local_workers = cfg.virtual_workers // ctx.world_size
    num_classes = num_classes_for_dataset(cfg.data.dataset, cfg.data.num_classes)
    inputs, targets = _random_packed_batch(
        cfg,
        device=device,
        local_workers=local_workers,
        num_classes=num_classes,
    )
    torch.backends.cudnn.benchmark = cfg.runtime.cudnn_benchmark
    configure_tf32(cfg.runtime, device)
    amp_enabled = cfg.runtime.amp and device.type in {"cpu", "cuda"}

    raw_model = cast(
        nn.Module,
        get_packed_model(
            cfg.model.name,
            num_classes=num_classes,
            num_models=local_workers,
        ).to(device),
    )
    raw_forward = cast(
        Callable[[torch.Tensor], torch.Tensor],
        torch.compile(raw_model, mode=cfg.runtime.compile_mode)
        if cfg.runtime.compile
        else raw_model,
    )
    baseline = _profile_phases(
        raw_model,
        raw_forward,
        inputs,
        targets,
        amp_enabled=amp_enabled,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
    )
    del raw_forward, raw_model
    _cleanup(device)

    trainer = build_trainer(cfg, ctx)
    assert trainer.model is not None and trainer.forward_model is not None
    trainer_model = cast(nn.Module, trainer.model)
    trainer_forward = _profile_phases(
        trainer_model,
        trainer.forward_model,
        inputs,
        targets,
        amp_enabled=trainer._amp_enabled(),
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
    )
    trainer_method = _profile_trainer_method(
        trainer,
        inputs,
        targets,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
    )
    return ForwardBackwardProfile(
        rank=ctx.rank,
        world_size=ctx.world_size,
        device=str(device),
        model=cfg.model.name.value,
        trainer=cfg.trainer.name,
        local_workers=local_workers,
        local_batch_size=cfg.data.batch_size,
        input_shape=tuple(inputs.shape),
        input_stride=tuple(inputs.stride()),
        input_dtype=str(inputs.dtype),
        channels_last=inputs.is_contiguous(memory_format=torch.channels_last),
        amp=trainer._amp_enabled(),
        compile=cfg.runtime.compile,
        compile_mode=cfg.runtime.compile_mode,
        cudnn_benchmark=cudnn_benchmark,
        cases={
            "packed_resnet_baseline": baseline,
            "trainer_forward_model": trainer_forward,
            "trainer_compute_local_gradients": trainer_method,
        },
    )


def _phase_median(case: ForwardBackwardCase, phase: str) -> str:
    stats = case.phases.get(phase)
    return f"{stats.median_ms:.3f}" if stats is not None else "n/a"


def _print_profile(profile: ForwardBackwardProfile) -> None:
    print(
        f"rank={profile.rank}/{profile.world_size} device={profile.device} "
        f"model={profile.model} trainer={profile.trainer} local_workers={profile.local_workers} "
        f"local_batch_size={profile.local_batch_size}"
    )
    print(
        f"amp={profile.amp} compile={profile.compile} compile_mode={profile.compile_mode} "
        f"cudnn_benchmark={profile.cudnn_benchmark}"
    )
    print(
        f"input_shape={profile.input_shape} stride={profile.input_stride} "
        f"dtype={profile.input_dtype} channels_last={profile.channels_last}"
    )
    print(
        f"{'case':<34} {'forward':>10} {'loss':>10} {'backward':>10} "
        f"{'total':>10} {'vs_base':>10} {'peak_MB':>10}"
    )
    baseline_ms = profile.cases["packed_resnet_baseline"].phases["total"].median_ms
    for name, case in profile.cases.items():
        total_ms = case.phases["total"].median_ms
        delta = 100.0 * (total_ms / baseline_ms - 1.0) if baseline_ms else 0.0
        peak = f"{case.peak_memory_mb:.1f}" if case.peak_memory_mb is not None else "n/a"
        print(
            f"{name:<34} {_phase_median(case, 'forward'):>10} "
            f"{_phase_median(case, 'loss'):>10} {_phase_median(case, 'backward'):>10} "
            f"{total_ms:>10.3f} {delta:>+9.1f}% {peak:>10}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    cfg = config_from_args(args)
    device = resolve_process_device(cfg.device)
    ctx = init_process_context(device)
    try:
        run_cfg = cfg.model_copy(update={"device": str(device), "epochs": max(cfg.epochs, 1)})
        profile = profile_forward_backward(
            run_cfg,
            ctx,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            cudnn_benchmark=args.cudnn_benchmark,
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
