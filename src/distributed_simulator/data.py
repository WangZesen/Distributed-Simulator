from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


class DatasetName(StrEnum):
    SYNTHETIC = "synthetic"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"


CIFAR_MEAN: dict[DatasetName, tuple[float, float, float]] = {
    DatasetName.CIFAR10: (0.4914, 0.4822, 0.4465),
    DatasetName.CIFAR100: (0.5071, 0.4867, 0.4408),
}

CIFAR_STD: dict[DatasetName, tuple[float, float, float]] = {
    DatasetName.CIFAR10: (0.2470, 0.2435, 0.2616),
    DatasetName.CIFAR100: (0.2675, 0.2565, 0.2761),
}


def num_classes_for_dataset(dataset: DatasetName, synthetic_num_classes: int) -> int:
    if dataset == DatasetName.CIFAR10:
        return 10
    if dataset == DatasetName.CIFAR100:
        return 100
    return synthetic_num_classes


class InMemorySyntheticImages:
    def __init__(
        self,
        samples: int,
        num_classes: int,
        seed: int,
        device: torch.device,
    ):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        images = torch.rand(samples, 3, 32, 32, generator=generator)
        teacher = torch.randn(3 * 32 * 32, num_classes, generator=generator)
        logits = images.flatten(start_dim=1) @ teacher
        labels = logits.argmax(dim=1)
        self.images = images.to(device=device, dtype=torch.float32)
        self.labels = labels.to(device=device, dtype=torch.long)
        self.device = device
        logger.info(
            "Loaded synthetic image data into {} memory: images={} labels={}",
            device,
            tuple(self.images.shape),
            tuple(self.labels.shape),
        )

    def __len__(self) -> int:
        return self.images.size(0)

    def batch_for_worker(
        self,
        worker_rank: int,
        virtual_workers: int,
        batch_size: int,
        epoch: int,
        step: int,
        seed: int,
        augment: bool,
    ) -> tuple[Tensor, Tensor]:
        del augment
        indices = deterministic_worker_indices(
            dataset_size=len(self),
            worker_rank=worker_rank,
            virtual_workers=virtual_workers,
            batch_size=batch_size,
            epoch=epoch,
            step=step,
            seed=seed,
            device=self.device,
            drop_last=True,
        )
        return self.images.index_select(0, indices), self.labels.index_select(0, indices)


class InMemoryCifar:
    def __init__(
        self,
        dataset: DatasetName,
        root: str | Path,
        train: bool,
        device: torch.device,
        download: bool = True,
    ):
        if dataset not in {DatasetName.CIFAR10, DatasetName.CIFAR100}:
            raise ValueError(f"expected CIFAR dataset, got {dataset}")
        images, labels = _load_cifar_tensors(dataset, root, train=train, download=download)
        self.dataset = dataset
        self.images = images.to(device=device, dtype=torch.float32).div_(255.0)
        self.labels = labels.to(device=device, dtype=torch.long)
        mean = torch.tensor(CIFAR_MEAN[dataset], device=device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD[dataset], device=device).view(1, 3, 1, 1)
        self.mean = mean
        self.std = std
        self.device = device
        self._epoch_orders: dict[tuple[int, int], Tensor] = {}
        logger.info(
            "Loaded {} split={} into {} memory: images={} labels={}",
            dataset.value,
            "train" if train else "test",
            device,
            tuple(self.images.shape),
            tuple(self.labels.shape),
        )

    def __len__(self) -> int:
        return self.images.size(0)

    def batch_for_worker(
        self,
        worker_rank: int,
        virtual_workers: int,
        batch_size: int,
        epoch: int,
        step: int,
        seed: int,
        augment: bool,
    ) -> tuple[Tensor, Tensor]:
        order = self.epoch_order(epoch=epoch, seed=seed)
        indices = worker_indices_from_order(
            order=order,
            worker_rank=worker_rank,
            virtual_workers=virtual_workers,
            batch_size=batch_size,
            step=step,
            drop_last=True,
        )
        images = self.images.index_select(0, indices)
        labels = self.labels.index_select(0, indices)
        if augment:
            images = deterministic_cifar_augment(
                images,
                seed=seed,
                epoch=epoch,
                step=step,
                worker_rank=worker_rank,
            )
        images = images.sub(self.mean).div(self.std)
        return images, labels

    def batch_for_workers(
        self,
        worker_ranks: Sequence[int],
        virtual_workers: int,
        batch_size: int,
        epoch: int,
        step: int,
        seed: int,
        augment: bool,
    ) -> tuple[Tensor, Tensor]:
        order = self.epoch_order(epoch=epoch, seed=seed)
        indices = torch.stack(
            [
                worker_indices_from_order(
                    order=order,
                    worker_rank=rank,
                    virtual_workers=virtual_workers,
                    batch_size=batch_size,
                    step=step,
                    drop_last=True,
                )
                for rank in worker_ranks
            ],
            dim=0,
        )
        local_workers = len(worker_ranks)
        images = self.images.index_select(0, indices.flatten())
        labels = self.labels.index_select(0, indices.flatten())
        images = images.view(local_workers, batch_size, *self.images.shape[1:])
        labels = labels.view(local_workers, batch_size)
        if augment:
            images = deterministic_cifar_augment_for_workers(
                images,
                seed=seed,
                epoch=epoch,
                step=step,
                worker_ranks=worker_ranks,
            )
        images = images.sub(self.mean).div(self.std)
        return images.transpose(0, 1).contiguous(), labels.transpose(0, 1).contiguous()

    def epoch_order(self, epoch: int, seed: int) -> Tensor:
        key = (epoch, seed)
        order = self._epoch_orders.get(key)
        if order is None:
            order = deterministic_epoch_order(
                dataset_size=len(self),
                epoch=epoch,
                seed=seed,
                device=self.device,
            )
            self._epoch_orders[key] = order
        return order


def deterministic_worker_indices(
    dataset_size: int,
    worker_rank: int,
    virtual_workers: int,
    batch_size: int,
    epoch: int,
    step: int,
    seed: int,
    device: torch.device,
    drop_last: bool,
) -> Tensor:
    if virtual_workers < 1 or virtual_workers & (virtual_workers - 1):
        raise ValueError("virtual_workers must be a positive power of two")
    if worker_rank < 0 or worker_rank >= virtual_workers:
        raise ValueError(f"worker_rank {worker_rank} is outside [0, {virtual_workers})")

    order = deterministic_epoch_order(
        dataset_size=dataset_size,
        epoch=epoch,
        seed=seed,
        device=device,
    )
    return worker_indices_from_order(
        order=order,
        worker_rank=worker_rank,
        virtual_workers=virtual_workers,
        batch_size=batch_size,
        step=step,
        drop_last=drop_last,
    )


def deterministic_epoch_order(
    dataset_size: int,
    epoch: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch * 1_000_003)
    return torch.randperm(dataset_size, generator=generator).to(device)


def worker_indices_from_order(
    order: Tensor,
    worker_rank: int,
    virtual_workers: int,
    batch_size: int,
    step: int,
    drop_last: bool,
) -> Tensor:
    if virtual_workers < 1 or virtual_workers & (virtual_workers - 1):
        raise ValueError("virtual_workers must be a positive power of two")
    if worker_rank < 0 or worker_rank >= virtual_workers:
        raise ValueError(f"worker_rank {worker_rank} is outside [0, {virtual_workers})")

    worker_order = order[worker_rank::virtual_workers]
    if drop_last:
        usable = (worker_order.numel() // batch_size) * batch_size
        worker_order = worker_order[:usable]
    if worker_order.numel() == 0:
        raise ValueError("worker shard is empty; reduce virtual_workers or batch_size")

    offset = (step * batch_size) % worker_order.numel()
    end = offset + batch_size
    if end <= worker_order.numel():
        return worker_order[offset:end]
    return torch.cat((worker_order[offset:], worker_order[: end % worker_order.numel()]))


def deterministic_cifar_augment(
    images: Tensor,
    seed: int,
    epoch: int,
    worker_rank: int | Tensor,
    step: int = 0,
    padding: int = 4,
    crop_size: int = 32,
    flip_p: float = 0.5,
) -> Tensor:
    batch, _, height, width = images.shape
    max_x = height + 2 * padding - crop_size + 1
    max_y = width + 2 * padding - crop_size + 1

    x_shifts, y_shifts, flip_mask = _cifar_augmentation_randomness(
        batch=batch,
        seed=seed,
        epoch=epoch,
        step=step,
        worker_rank=worker_rank,
        max_x=max_x,
        max_y=max_y,
        flip_p=flip_p,
        device=images.device,
    )
    return _apply_cifar_augment(
        images,
        x_shifts=x_shifts,
        y_shifts=y_shifts,
        flip_mask=flip_mask,
        padding=padding,
        crop_size=crop_size,
    )


def deterministic_cifar_augment_for_workers(
    images: Tensor,
    seed: int,
    epoch: int,
    worker_ranks: Sequence[int],
    step: int = 0,
    padding: int = 4,
    crop_size: int = 32,
    flip_p: float = 0.5,
) -> Tensor:
    if images.dim() != 5:
        raise ValueError("expected images with shape (workers, batch, channels, height, width)")
    local_workers, batch, channels, height, width = images.shape
    if local_workers != len(worker_ranks):
        raise ValueError("worker_ranks length must match images.size(0)")

    max_x = height + 2 * padding - crop_size + 1
    max_y = width + 2 * padding - crop_size + 1
    random_params = [
        _cifar_augmentation_randomness(
            batch=batch,
            seed=seed,
            epoch=epoch,
            step=step,
            worker_rank=rank,
            max_x=max_x,
            max_y=max_y,
            flip_p=flip_p,
            device=images.device,
        )
        for rank in worker_ranks
    ]
    x_shifts = torch.stack([params[0] for params in random_params], dim=0).flatten()
    y_shifts = torch.stack([params[1] for params in random_params], dim=0).flatten()
    flip_mask = torch.stack([params[2] for params in random_params], dim=0).flatten()
    augmented = _apply_cifar_augment(
        images.reshape(local_workers * batch, channels, height, width),
        x_shifts=x_shifts,
        y_shifts=y_shifts,
        flip_mask=flip_mask,
        padding=padding,
        crop_size=crop_size,
    )
    return augmented.view(local_workers, batch, channels, crop_size, crop_size)


def _cifar_augmentation_randomness(
    *,
    batch: int,
    seed: int,
    epoch: int,
    step: int,
    worker_rank: int | Tensor,
    max_x: int,
    max_y: int,
    flip_p: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    if isinstance(worker_rank, int):
        worker_offset = worker_rank * 10_007
    else:
        worker_offset = 0
    generator.manual_seed(seed + epoch * 1_000_003 + step * 100_003 + worker_offset)
    x_shifts = torch.randint(max_x, (batch,), generator=generator).to(device)
    y_shifts = torch.randint(max_y, (batch,), generator=generator).to(device)
    flip_mask = torch.rand(batch, generator=generator).to(device) < flip_p
    if isinstance(worker_rank, Tensor):
        x_shifts = (x_shifts + worker_rank.to(device=device) * 3) % max_x
        y_shifts = (y_shifts + worker_rank.to(device=device) * 5) % max_y
    return x_shifts, y_shifts, flip_mask


def _apply_cifar_augment(
    images: Tensor,
    *,
    x_shifts: Tensor,
    y_shifts: Tensor,
    flip_mask: Tensor,
    padding: int,
    crop_size: int,
) -> Tensor:
    batch = images.size(0)
    padded = F.pad(images, (padding, padding, padding, padding))
    batch_index = torch.arange(batch, device=images.device).view(batch, 1, 1)
    y_index = torch.arange(crop_size, device=images.device).view(1, crop_size, 1) + x_shifts.view(
        batch, 1, 1
    )
    x_index = torch.arange(crop_size, device=images.device).view(1, 1, crop_size) + y_shifts.view(
        batch, 1, 1
    )
    augmented = padded[batch_index, :, y_index, x_index].permute(0, 3, 1, 2).contiguous()
    flipped = augmented.flip(dims=(3,))
    return torch.where(flip_mask.view(batch, 1, 1, 1), flipped, augmented)


def _load_cifar_tensors(
    dataset: DatasetName,
    root: str | Path,
    train: bool,
    download: bool,
) -> tuple[Tensor, Tensor]:
    try:
        from torchvision.datasets import CIFAR10, CIFAR100
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise RuntimeError("CIFAR support requires torchvision to be installed") from exc

    dataset_cls = CIFAR10 if dataset == DatasetName.CIFAR10 else CIFAR100
    raw = dataset_cls(root=str(root), train=train, download=download)
    images = torch.from_numpy(raw.data).permute(0, 3, 1, 2).contiguous()
    labels = torch.tensor(raw.targets, dtype=torch.long)
    return images, labels
