from __future__ import annotations

import argparse
import sys

import torch
from loguru import logger

from distributed_simulator.config import (
    ConstantSchedulerConfig,
    DataConfig,
    DecentralizedConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    Topology,
    WarmupCosineSchedulerConfig,
)
from distributed_simulator.data import DatasetName
from distributed_simulator.distributed import destroy_process_context, init_process_context
from distributed_simulator.model import ModelName
from distributed_simulator.trainer import DecentralizedTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simulated decentralized training.")
    parser.add_argument("--workers", type=int, default=8, help="Number of virtual workers.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument(
        "--topology",
        choices=[item.value for item in Topology],
        default=Topology.RING.value,
    )
    parser.add_argument("--device", default="cpu", help="Torch device for local tensors.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--scheduler", choices=["constant", "warmup_cosine"], default="warmup_cosine"
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--eta-min-factor", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--model", choices=[item.value for item in ModelName], default=ModelName.WRN_16_8.value
    )
    parser.add_argument(
        "--dataset",
        choices=[item.value for item in DatasetName],
        default=DatasetName.CIFAR10.value,
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=10000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model forward path with torch.compile.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile optimization mode.",
    )
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--log-level", default="INFO", help="Loguru log level.")
    return parser


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), enqueue=True)


def config_from_args(args: argparse.Namespace) -> DecentralizedConfig:
    data = DataConfig(
        dataset=DatasetName(args.dataset),
        root=args.data_root,
        download=not args.no_download,
        augment=not args.no_augment,
        num_classes=args.classes,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed + 10_000,
    )
    model = ModelConfig(name=ModelName(args.model))
    optimizer = OptimizerConfig(
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        ConstantSchedulerConfig()
        if args.scheduler == "constant"
        else WarmupCosineSchedulerConfig(
            warmup_epochs=args.warmup_epochs,
            warmup_start_factor=args.warmup_start_factor,
            eta_min_factor=args.eta_min_factor,
        )
    )
    runtime = RuntimeConfig(
        amp=not args.no_amp,
        amp_dtype="bf16",
        compile=args.compile,
        compile_mode=args.compile_mode,
    )
    return DecentralizedConfig(
        virtual_workers=args.workers,
        topology=Topology(args.topology),
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        runtime=runtime,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    cfg = config_from_args(args)
    device = torch.device(cfg.device)
    ctx = init_process_context(device)
    try:
        run_cfg = cfg.model_copy(update={"device": str(device)})
        if ctx.rank == 0:
            logger.info(
                "Launching decentralized simulation: "
                "workers={} processes={} topology={} model={} dataset={} device={} epochs={}",
                run_cfg.virtual_workers,
                ctx.world_size,
                run_cfg.topology.value,
                run_cfg.model.name.value,
                run_cfg.data.dataset.value,
                run_cfg.device,
                run_cfg.epochs,
            )
        trainer = DecentralizedTrainer(run_cfg, ctx)
        metrics = trainer.train()
        if ctx.rank == 0:
            logger.info(
                "Finished decentralized simulation: loss={:.6f} d2c={:.6f}",
                metrics.loss,
                metrics.distance_to_consensus,
            )
            print(
                "decentralized "
                f"workers={cfg.virtual_workers} processes={ctx.world_size} "
                f"topology={cfg.topology.value} epochs={metrics.epochs} steps={metrics.steps} "
                f"loss={metrics.loss:.6f} d2c={metrics.distance_to_consensus:.6f}"
            )
    finally:
        destroy_process_context()


if __name__ == "__main__":
    main()
