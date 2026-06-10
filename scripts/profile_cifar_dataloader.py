from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from packed_resnet import create_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Packed-ResNet's CIFAR dataloader.")
    parser.add_argument(
        "--datasets", nargs="+", choices=["CIFAR10", "CIFAR100"], default=["CIFAR10", "CIFAR100"]
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--virtual-workers", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers_per_process = args.virtual_workers // args.world_size
    start = args.rank * workers_per_process
    owned_ranks = tuple(range(start, start + workers_per_process))
    device = torch.device(args.device)

    for dataset in args.datasets:
        start_time = perf_counter()
        loader = create_dataloader(
            dataset.lower(),
            root=args.root,
            local_batch_size=args.batch_size,
            world_size=args.virtual_workers,
            ranks=owned_ranks,
            base_seed=args.seed,
            train=True,
            packed=True,
            channels_last=True,
            augment=not args.no_augment,
            device=device,
            sampler_drop_last=True,
            drop_last=True,
        )
        loader.set_epoch(args.epoch)
        load_seconds = perf_counter() - start_time
        synchronize(device)
        start_time = perf_counter()
        batches = tuple(loader)
        synchronize(device)
        elapsed = perf_counter() - start_time
        images, targets = batches[0]
        samples = len(batches) * args.batch_size * len(owned_ranks)
        print(
            f"{dataset}: load={load_seconds:.3f}s epoch={elapsed:.3f}s "
            f"samples/s={samples / elapsed:.1f} images={tuple(images.shape)} "
            f"targets={tuple(targets.shape)} channels_last="
            f"{images.is_contiguous(memory_format=torch.channels_last)}"
        )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
