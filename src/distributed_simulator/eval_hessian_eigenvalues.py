from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from packed_resnet import WideResNet, create_dataloader

from distributed_simulator.config import SimulationConfig, load_config_files
from distributed_simulator.data import (
    DatasetName,
    InMemorySyntheticImages,
    num_classes_for_dataset,
)
from distributed_simulator.distributed import (
    ProcessContext,
    destroy_process_context,
    init_process_context,
    resolve_process_device,
)
from distributed_simulator.hessian_eigenvalues import (
    HvpBatchSize,
    LanczosEigenvalueEstimate,
    lanczos_eigenvalues,
)
from distributed_simulator.logging import LOG_FORMAT
from distributed_simulator.model import ModelName, get_model

DEFAULT_LANCZOS_SEED = 456
DEFAULT_NUM_EIGENVALUES = 30
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_BN_CALIBRATION = "none"
CALIBRATION_EPOCH = 123321
EVALUATION_EPOCH = 123321


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate smallest and largest Hessian eigenvalues for a saved checkpoint.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint path; defaults to RUN_DIR/checkpoints/global_last.pth.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-eigenvalues", type=int, default=DEFAULT_NUM_EIGENVALUES)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument(
        "--bn-calibration",
        choices=["none", "clean", "augmented"],
        default=DEFAULT_BN_CALIBRATION,
        help=(
            "Batch norm buffer handling before HVP evaluation: use checkpoint buffers, "
            "recalibrate on clean training data, or recalibrate on augmented training data."
        ),
    )
    parser.add_argument(
        "--eval-train-mode",
        action="store_true",
        help="Run HVP evaluation with the model in train mode. Requires --bn-calibration=none.",
    )
    parser.add_argument(
        "--augment-eval-data",
        action="store_true",
        help="Use deterministic training augmentation for the data used in Hessian evaluation.",
    )
    parser.add_argument("--device", help="Override device from run config.")
    parser.add_argument("--seed", type=int, default=DEFAULT_LANCZOS_SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser


def configure_logging(level: str, ctx: ProcessContext) -> None:
    logger.remove()
    if ctx.rank == 0:
        logger.add(sys.stderr, level=level.upper(), format=LOG_FORMAT, enqueue=True)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    cfg = _load_run_config(args.run_dir)
    device = resolve_process_device(args.device or cfg.device)
    ctx = init_process_context(device)
    configure_logging(args.log_level, ctx)

    try:
        cfg = cfg.model_copy(update={"device": str(device)})
        checkpoint = args.checkpoint or args.run_dir / "checkpoints" / "global_last.pth"
        model = _load_model(cfg, checkpoint, device)
        _prepare_model_buffers_and_mode(
            model,
            cfg,
            device,
            ctx,
            batch_size=args.eval_batch_size,
            bn_calibration=args.bn_calibration,
            eval_train_mode=args.eval_train_mode,
        )
        full_batch = _evaluation_batch(
            cfg,
            device,
            ctx,
            data_fraction=args.data_fraction,
            augment=args.augment_eval_data,
        )
        batch_size = HvpBatchSize(args.eval_batch_size)
        logger.info(
            "Estimating Hessian eigenvalues: checkpoint={} samples={} batch_size={} "
            "iters={} bn_calibration={} eval_data={} mode={}",
            checkpoint,
            full_batch[0].size(0),
            batch_size.value,
            args.num_eigenvalues,
            args.bn_calibration,
            "augmented" if args.augment_eval_data else "clean",
            "train" if args.eval_train_mode else "eval",
        )
        estimate = lanczos_eigenvalues(
            model,
            full_batch,
            batch_size,
            num_iters=args.num_eigenvalues,
            seed=args.seed,
        )
        if ctx.rank == 0:
            output_path = args.run_dir / "hessian_eigenvalues.csv"
            _append_result(
                output_path,
                checkpoint=checkpoint,
                eval_batch_size=batch_size.value,
                num_eigenvalues=args.num_eigenvalues,
                data_fraction=args.data_fraction,
                bn_calibration=args.bn_calibration,
                eval_train_mode=args.eval_train_mode,
                augment_eval_data=args.augment_eval_data,
                estimate=estimate,
            )
            print(
                "hessian_eigenvalues "
                f"checkpoint={checkpoint} "
                f"lambda_min={estimate.lambda_min:.8g} "
                f"lambda_max={estimate.lambda_max:.8g} "
                f"output={output_path}"
            )
    finally:
        destroy_process_context()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.run_dir.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {args.run_dir}")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive")
    if args.num_eigenvalues < 1:
        raise ValueError("--num-eigenvalues must be positive")
    if not 0.0 < args.data_fraction <= 1.0:
        raise ValueError("--data-fraction must be in (0, 1]")
    if args.eval_train_mode and args.bn_calibration != "none":
        raise ValueError("--eval-train-mode requires --bn-calibration=none")


def _load_run_config(run_dir: Path) -> SimulationConfig:
    config_path = run_dir / "config.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"run directory does not contain config.toml: {run_dir}")
    return load_config_files([config_path])


def _load_model(
    cfg: SimulationConfig,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    num_classes = num_classes_for_dataset(cfg.data.dataset, cfg.data.num_classes)
    model = _checkpoint_model(cfg, num_classes=num_classes).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _checkpoint_model(cfg: SimulationConfig, *, num_classes: int) -> torch.nn.Module:
    if cfg.model.name == ModelName.LINEAR:
        return get_model(cfg.model.name, num_classes=num_classes)
    depth = int(cfg.model.name.value.split("_")[1])
    widen_factor = int(cfg.model.name.value.split("_")[-1])
    return WideResNet(depth=depth, widen_factor=widen_factor, num_classes=num_classes)


def _prepare_model_buffers_and_mode(
    model: torch.nn.Module,
    cfg: SimulationConfig,
    device: torch.device,
    ctx: ProcessContext,
    *,
    batch_size: int,
    bn_calibration: str,
    eval_train_mode: bool,
) -> None:
    if eval_train_mode:
        model.train()
        return
    if bn_calibration == "none":
        model.eval()
        return
    _calibrate_batch_norm(
        model,
        cfg,
        device,
        ctx,
        batch_size=batch_size,
        augment=bn_calibration == "augmented",
    )


def _batch_norm_layers(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [
        module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


@torch.no_grad()
def _calibrate_batch_norm(
    model: nn.Module,
    cfg: SimulationConfig,
    device: torch.device,
    ctx: ProcessContext,
    *,
    batch_size: int,
    augment: bool,
) -> None:
    layers = _batch_norm_layers(model)
    if not layers:
        model.eval()
        return

    for layer in layers:
        layer.reset_running_stats()

    model.train()
    if cfg.data.dataset == DatasetName.SYNTHETIC:
        images, _, _ = _calibration_images(cfg, device, ctx)
        batches = (
            images[start : start + batch_size] for start in range(0, images.size(0), batch_size)
        )
    else:
        loader = _create_cifar_loader(
            cfg,
            device,
            ctx,
            batch_size=batch_size,
            augment=augment,
        )
        loader.set_epoch(CALIBRATION_EPOCH)
        batches = (images for images, _ in loader)
    for batch in batches:
        model(batch)

    if ctx.is_distributed:
        for layer in layers:
            if layer.running_mean is not None:
                dist.all_reduce(layer.running_mean, op=dist.ReduceOp.SUM)
                layer.running_mean.div_(ctx.world_size)
            if layer.running_var is not None:
                dist.all_reduce(layer.running_var, op=dist.ReduceOp.SUM)
                layer.running_var.div_(ctx.world_size)
            if layer.num_batches_tracked is not None:
                dist.all_reduce(layer.num_batches_tracked, op=dist.ReduceOp.MAX)

    model.eval()


def _calibration_images(
    cfg: SimulationConfig,
    device: torch.device,
    ctx: ProcessContext,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if cfg.data.dataset == DatasetName.SYNTHETIC:
        dataset = InMemorySyntheticImages(
            samples=cfg.virtual_workers * cfg.data.batch_size * 4,
            num_classes=cfg.data.num_classes,
            seed=cfg.data.seed,
            device=device,
        )
        images = dataset.images
        mean = None
        std = None
    else:
        loader = _create_cifar_loader(cfg, device, ctx, batch_size=cfg.data.eval_batch_size)
        loader.set_epoch(CALIBRATION_EPOCH)
        images = torch.cat([batch for batch, _ in loader])
        mean = None
        std = None

    if ctx.is_distributed and cfg.data.dataset == DatasetName.SYNTHETIC:
        images = images[ctx.rank :: ctx.world_size].contiguous()
    if images.size(0) == 0:
        raise ValueError("this process received no calibration samples")
    return images, mean, std


def _evaluation_batch(
    cfg: SimulationConfig,
    device: torch.device,
    ctx: ProcessContext,
    *,
    data_fraction: float,
    augment: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.data.dataset == DatasetName.SYNTHETIC:
        dataset = InMemorySyntheticImages(
            samples=cfg.virtual_workers * cfg.data.batch_size * 4,
            num_classes=cfg.data.num_classes,
            seed=cfg.data.seed,
            device=device,
        )
        images = dataset.images
        labels = dataset.labels
    else:
        loader = _create_cifar_loader(
            cfg,
            device,
            ctx,
            batch_size=cfg.data.eval_batch_size,
            augment=augment,
        )
        loader.set_epoch(EVALUATION_EPOCH)
        batches = tuple(loader)
        images = torch.cat([batch_images for batch_images, _ in batches])
        labels = torch.cat([batch_labels for _, batch_labels in batches])

    if cfg.data.dataset == DatasetName.SYNTHETIC:
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.data.seed)
        order = torch.randperm(images.size(0), generator=generator, device=device)
        images = images.index_select(0, order)
        labels = labels.index_select(0, order)

    selected = int(images.size(0) * data_fraction)
    if selected < 1:
        raise ValueError("--data-fraction selects no samples")
    images = images[:selected]
    labels = labels[:selected]
    if ctx.is_distributed and cfg.data.dataset == DatasetName.SYNTHETIC:
        images = images[ctx.rank :: ctx.world_size].contiguous()
        labels = labels[ctx.rank :: ctx.world_size].contiguous()
    if images.size(0) == 0:
        raise ValueError("this process received no HVP samples; increase --data-fraction")
    if images.ndim == 4:
        images = images.contiguous(memory_format=torch.channels_last)
    return images, labels


def _create_cifar_loader(
    cfg: SimulationConfig,
    device: torch.device,
    ctx: ProcessContext,
    *,
    batch_size: int,
    augment: bool = False,
):
    return create_dataloader(
        cfg.data.dataset.value.lower(),
        root=cfg.data.root,
        local_batch_size=batch_size,
        world_size=ctx.world_size,
        ranks=[ctx.rank],
        base_seed=cfg.data.seed,
        train=True,
        packed=False,
        channels_last=True,
        shuffle=True,
        augment=augment,
        device=device,
    )


def _append_result(
    path: Path,
    *,
    checkpoint: Path,
    eval_batch_size: int,
    num_eigenvalues: int,
    data_fraction: float,
    bn_calibration: str,
    eval_train_mode: bool,
    augment_eval_data: bool,
    estimate: LanczosEigenvalueEstimate,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = (
        "checkpoint",
        "eval_batch_size",
        "num_eigenvalues",
        "data_fraction",
        "bn_calibration",
        "eval_train_mode",
        "augment_eval_data",
        "lambda_min",
        "lambda_max",
        "eigenvalues",
    )
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "checkpoint": str(checkpoint),
                "eval_batch_size": eval_batch_size,
                "num_eigenvalues": num_eigenvalues,
                "data_fraction": data_fraction,
                "bn_calibration": bn_calibration,
                "eval_train_mode": eval_train_mode,
                "augment_eval_data": augment_eval_data,
                "lambda_min": estimate.lambda_min,
                "lambda_max": estimate.lambda_max,
                "eigenvalues": json.dumps(list(estimate.eigenvalues)),
            }
        )


if __name__ == "__main__":
    main()
