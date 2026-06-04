from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from distributed_simulator.config import (
    DecentralizedTrainerConfig,
    SimulationConfig,
    SyncTrainerConfig,
    config_from_files_and_overrides,
)
from distributed_simulator.data import DatasetName
from distributed_simulator.distributed import (
    destroy_process_context,
    init_process_context,
    resolve_process_device,
)
from distributed_simulator.model import ModelName
from distributed_simulator.trainers import DecentralizedTrainer, SyncTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simulated distributed training.")
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="TOML config files loaded in order; later files override earlier files.",
    )
    parser.add_argument("--workers", type=int, help="Override number of virtual workers.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--device", help="Override torch device for local tensors.")
    parser.add_argument("--seed", type=int, help="Override base seed.")
    parser.add_argument(
        "--model",
        choices=[item.value for item in ModelName],
        help="Override model name.",
    )
    parser.add_argument(
        "--dataset",
        choices=[item.value for item in DatasetName],
        help="Override dataset name.",
    )
    parser.add_argument("--batch-size", type=int, help="Override per-worker training batch size.")
    parser.add_argument("--classes", type=int, help="Override synthetic class count.")
    parser.add_argument("--log-level", default="INFO", help="Loguru log level.")
    return parser


def configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), enqueue=True)


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return config_from_files_and_overrides(args.configs, _overrides_from_args(args))


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.workers is not None:
        overrides["virtual_workers"] = args.workers
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.model is not None:
        overrides["model"] = {"name": args.model}

    data: dict[str, Any] = {}
    if args.dataset is not None:
        data["dataset"] = args.dataset
    if args.batch_size is not None:
        data["batch_size"] = args.batch_size
    if args.classes is not None:
        data["num_classes"] = args.classes
    if data:
        overrides["data"] = data
    return overrides


def build_trainer(cfg: SimulationConfig, ctx):  # noqa: ANN001, ANN201
    trainer_cfg = cfg.trainer
    if isinstance(trainer_cfg, DecentralizedTrainerConfig):
        return DecentralizedTrainer(cfg, ctx)
    if isinstance(trainer_cfg, SyncTrainerConfig):
        return SyncTrainer(cfg, ctx)
    raise ValueError(f"CLI does not support trainer: {trainer_cfg.name}")


def _trainer_summary_fields(trainer_cfg: object) -> str:
    if isinstance(trainer_cfg, DecentralizedTrainerConfig):
        return f"topology={trainer_cfg.topology.value} mix={trainer_cfg.mix.name} "
    return ""


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    cfg = config_from_args(args)
    device = resolve_process_device(cfg.device)
    ctx = init_process_context(device)
    try:
        run_cfg = cfg.model_copy(update={"device": str(device)})
        trainer_cfg = run_cfg.trainer
        if ctx.rank == 0:
            logger.info(
                "Launching {} simulation: "
                "workers={} processes={} model={} dataset={} "
                "device={} epochs={}",
                trainer_cfg.name,
                run_cfg.virtual_workers,
                ctx.world_size,
                run_cfg.model.name.value,
                run_cfg.data.dataset.value,
                run_cfg.device,
                run_cfg.epochs,
            )
        trainer = build_trainer(run_cfg, ctx)
        metrics = trainer.train()
        if ctx.rank == 0:
            logger.info(
                "Finished {} simulation: loss={:.6f} d2c={:.6f}",
                trainer_cfg.name,
                metrics.loss,
                metrics.distance_to_consensus,
            )
            details = _trainer_summary_fields(trainer_cfg)
            print(
                f"{trainer_cfg.name} "
                f"workers={run_cfg.virtual_workers} processes={ctx.world_size} "
                f"{details}"
                f"epochs={metrics.epochs} "
                f"steps={metrics.steps} "
                f"loss={metrics.loss:.6f} d2c={metrics.distance_to_consensus:.6f} "
                f"gamma={metrics.gamma:.6f} accum_gamma={metrics.accumulated_gamma:.6f}"
            )
    finally:
        destroy_process_context()


if __name__ == "__main__":
    main()
