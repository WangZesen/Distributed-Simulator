from __future__ import annotations

import argparse
import cProfile
import math
import pstats
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from distributed_simulator.data import DatasetName, InMemoryCifar


@dataclass(frozen=True)
class TimingSummary:
    dataset: DatasetName
    load_seconds: float
    measured_batches: int
    total_seconds: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    samples_per_second: float
    batch_shape: tuple[int, ...]
    label_shape: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the actual in-memory CIFAR loader used by distributed_simulator training."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[DatasetName.CIFAR10.value, DatasetName.CIFAR100.value],
        default=[DatasetName.CIFAR10.value, DatasetName.CIFAR100.value],
        help="CIFAR datasets to profile.",
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="CIFAR data root.")
    parser.add_argument("--device", default="cpu", help="Target device used by InMemoryCifar.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-virtual-worker batch size.")
    parser.add_argument(
        "--virtual-workers",
        type=int,
        default=8,
        help="Total simulated worker count.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Simulated launched process count for local-worker batching.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Simulated process rank. Only this rank's owned virtual workers are profiled.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch index used for deterministic sampling.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Data sampling seed.")
    parser.add_argument("--steps", type=int, default=128, help="Number of measured training steps.")
    parser.add_argument("--warmup-steps", type=int, default=64, help="Unmeasured warmup steps.")
    parser.add_argument(
        "--full-epoch",
        action="store_true",
        help="Measure all batches in one worker shard instead of --steps.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable deterministic CIFAR crop/flip augmentation.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require CIFAR files to already exist under --root.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Optional cProfile output path for the measured batch loop.",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=25,
        help="Print this many cProfile rows when --profile-output is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    _validate_positive_power_of_two(args.virtual_workers, "virtual-workers")
    _validate_positive_power_of_two(args.world_size, "world-size")
    if args.virtual_workers % args.world_size:
        raise ValueError("--virtual-workers must be divisible by --world-size")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank must be in [0, --world-size)")

    owned_ranks = _owned_ranks(args.virtual_workers, args.world_size, args.rank)
    print(
        "Profiling CIFAR loader with "
        f"device={device}, batch_size={args.batch_size}, "
        f"virtual_workers={args.virtual_workers}, owned_ranks={owned_ranks}, "
        f"augment={not args.no_augment}"
    )
    print()

    for dataset_value in args.datasets:
        summary = profile_dataset(
            dataset=DatasetName(dataset_value),
            root=args.root,
            device=device,
            download=not args.no_download,
            augment=not args.no_augment,
            virtual_workers=args.virtual_workers,
            owned_ranks=owned_ranks,
            batch_size=args.batch_size,
            epoch=args.epoch,
            seed=args.seed,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            full_epoch=args.full_epoch,
            profile_output=args.profile_output,
            profile_top=args.profile_top,
        )
        print_summary(summary)
        print()


def profile_dataset(
    *,
    dataset: DatasetName,
    root: Path,
    device: torch.device,
    download: bool,
    augment: bool,
    virtual_workers: int,
    owned_ranks: tuple[int, ...],
    batch_size: int,
    epoch: int,
    seed: int,
    steps: int,
    warmup_steps: int,
    full_epoch: bool,
    profile_output: Path | None,
    profile_top: int,
) -> TimingSummary:
    synchronize(device)
    start = perf_counter()
    loader = InMemoryCifar(dataset, root=root, train=True, device=device, download=download)
    synchronize(device)
    load_seconds = perf_counter() - start

    batches_per_epoch = max((len(loader) // virtual_workers) // batch_size, 1)
    measured_steps = batches_per_epoch if full_epoch else steps
    measured_steps = max(measured_steps, 1)

    with torch.no_grad():
        for step in range(max(warmup_steps, 0)):
            _batch_from_loader(
                loader,
                owned_ranks=owned_ranks,
                virtual_workers=virtual_workers,
                batch_size=batch_size,
                epoch=epoch,
                step=step,
                seed=seed,
                augment=augment,
            )
        synchronize(device)

        batch_times: list[float] = []
        first_batch_shape: tuple[int, ...] | None = None
        first_label_shape: tuple[int, ...] | None = None

        profiler = cProfile.Profile() if profile_output is not None else None
        if profiler is not None:
            profiler.enable()

        total_start = perf_counter()
        for step in range(measured_steps):
            step_start = perf_counter()
            images, labels = _batch_from_loader(
                loader,
                owned_ranks=owned_ranks,
                virtual_workers=virtual_workers,
                batch_size=batch_size,
                epoch=epoch,
                step=step,
                seed=seed,
                augment=augment,
            )
            synchronize(device)
            batch_times.append(perf_counter() - step_start)
            if first_batch_shape is None:
                first_batch_shape = tuple(images.shape)
                first_label_shape = tuple(labels.shape)
        total_seconds = perf_counter() - total_start

        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(profile_output))
            print(f"Wrote cProfile stats for {dataset.value} to {profile_output}")
            pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(profile_top)

    samples = measured_steps * len(owned_ranks) * batch_size
    return TimingSummary(
        dataset=dataset,
        load_seconds=load_seconds,
        measured_batches=measured_steps,
        total_seconds=total_seconds,
        mean_ms=1000.0 * (sum(batch_times) / len(batch_times)),
        median_ms=1000.0 * percentile(batch_times, 0.50),
        p95_ms=1000.0 * percentile(batch_times, 0.95),
        min_ms=1000.0 * min(batch_times),
        max_ms=1000.0 * max(batch_times),
        samples_per_second=samples / total_seconds,
        batch_shape=first_batch_shape or (),
        label_shape=first_label_shape or (),
    )


def _batch_from_loader(
    loader: InMemoryCifar,
    *,
    owned_ranks: tuple[int, ...],
    virtual_workers: int,
    batch_size: int,
    epoch: int,
    step: int,
    seed: int,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return loader.batch_for_workers(
        worker_ranks=owned_ranks,
        virtual_workers=virtual_workers,
        batch_size=batch_size,
        epoch=epoch,
        step=step,
        seed=seed,
        augment=augment,
    )


def _owned_ranks(virtual_workers: int, world_size: int, rank: int) -> tuple[int, ...]:
    workers_per_process = virtual_workers // world_size
    start = rank * workers_per_process
    return tuple(range(start, start + workers_per_process))


def _validate_positive_power_of_two(value: int, name: str) -> None:
    if value < 1 or value & (value - 1):
        raise ValueError(f"--{name} must be a positive power of two")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    index = min(math.ceil(q * len(ordered)) - 1, len(ordered) - 1)
    return ordered[index]


def print_summary(summary: TimingSummary) -> None:
    rows: list[tuple[str, Callable[[], object]]] = [
        ("dataset", lambda: summary.dataset.value),
        ("load_seconds", lambda: f"{summary.load_seconds:.4f}"),
        ("measured_batches", lambda: summary.measured_batches),
        ("total_batch_seconds", lambda: f"{summary.total_seconds:.4f}"),
        ("mean_batch_ms", lambda: f"{summary.mean_ms:.3f}"),
        ("median_batch_ms", lambda: f"{summary.median_ms:.3f}"),
        ("p95_batch_ms", lambda: f"{summary.p95_ms:.3f}"),
        ("min_batch_ms", lambda: f"{summary.min_ms:.3f}"),
        ("max_batch_ms", lambda: f"{summary.max_ms:.3f}"),
        ("samples_per_second", lambda: f"{summary.samples_per_second:.1f}"),
        ("batch_shape", lambda: summary.batch_shape),
        ("label_shape", lambda: summary.label_shape),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:{width}}  {value()}")


if __name__ == "__main__":
    main()
