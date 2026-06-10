from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, Self, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from packed_resnet import PackedDataLoader, create_dataloader

from distributed_simulator.config import SimulationConfig
from distributed_simulator.data import (
    DatasetName,
    InMemorySyntheticImages,
    num_classes_for_dataset,
)
from distributed_simulator.distributed import ProcessContext, all_gather_owned_buckets
from distributed_simulator.model import (
    PackedParameterLayout,
    get_packed_model,
    parameter_storage_layout,
)
from distributed_simulator.optim import NORM_MODULES, parameter_decay_mask
from distributed_simulator.parameters import average_distance_to_consensus
from distributed_simulator.scheduler import lr_factor

_FOREACH_ADD = "_foreach_add"
_FOREACH_ADD_INPLACE = "_foreach_add_"
_FOREACH_MUL_INPLACE = "_foreach_mul_"
_EVALUATION_INTERVAL_EPOCHS = 5


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    test_accuracy: float
    distance_to_consensus: float
    lr: float
    gamma: float
    accumulated_gamma: float


@dataclass(frozen=True)
class TrainMetrics:
    loss: float
    distance_to_consensus: float
    test_loss: float
    test_accuracy: float
    lr: float
    gamma: float
    accumulated_gamma: float
    epochs: int
    steps: int
    rank: int
    world_size: int
    owned_workers: tuple[int, ...]
    history: tuple[EpochMetrics, ...]


@dataclass
class PrefetchedBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    stream: torch.cuda.Stream


class PackedModel(Protocol):
    parameter_storage: torch.Tensor
    training: bool

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def train(self, mode: bool = True) -> Self: ...

    def eval(self) -> Self: ...

    def to(self, *args: object, **kwargs: object) -> Self: ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def modules(self) -> Iterator[nn.Module]: ...

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]: ...

    def named_buffers(self) -> Iterator[tuple[str, torch.Tensor]]: ...

    def sync_storage_from_parameters_(self) -> Self: ...

    def sync_parameters_from_storage_(self) -> Self: ...


class CudaBatchPrefetcher:
    def __init__(self, trainer: BaseTrainer):
        self.trainer = trainer
        self.stream = torch.cuda.Stream(device=trainer.device)

    def prefetch(self, step: int) -> PrefetchedBatch:
        with torch.cuda.stream(self.stream):
            inputs, targets = self.trainer._batch_for_training_step(step)
        return PrefetchedBatch(inputs=inputs, targets=targets, stream=self.stream)

    def wait(self, batch: PrefetchedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        current_stream = torch.cuda.current_stream(self.trainer.device)
        current_stream.wait_stream(batch.stream)
        batch.inputs.record_stream(current_stream)
        batch.targets.record_stream(current_stream)
        return batch.inputs, batch.targets


class BaseTrainer:
    def __init__(self, cfg: SimulationConfig, trainer_cfg: Any, ctx: ProcessContext | None = None):
        self.cfg = cfg
        self.trainer_cfg = trainer_cfg
        self.device = torch.device(cfg.device)
        self.ctx = ctx or ProcessContext()
        self._validate_process_layout()

        self.owned_ranks = self._owned_ranks()
        self.workers_per_process = self.cfg.virtual_workers // self.ctx.world_size
        self.local_worker_count = len(self.owned_ranks)
        self.local_index_by_rank = {rank: i for i, rank in enumerate(self.owned_ranks)}
        self.dataset = self._init_data(train=True)
        self.test_dataset = self._init_data(train=False)
        self._epoch_batch_cache: dict[
            int, tuple[int, tuple[tuple[torch.Tensor, torch.Tensor], ...]]
        ] = {}
        self.training_step = 0
        self.batches_per_epoch = self._batches_per_epoch()
        self.test_batch_size = self._test_batch_size()
        self.test_batches_per_epoch = self._test_batches_per_epoch()
        self.total_steps = self.cfg.epochs * self.batches_per_epoch
        self.model: PackedModel | None = None
        self._model_forward: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.param_storage: torch.Tensor | None = None
        self.param_layout: tuple[PackedParameterLayout, ...] = ()
        self.decay_parameters: list[torch.Tensor] = []
        self.no_decay_parameters: list[torch.Tensor] = []
        self.active_decay_parameters: list[torch.Tensor] = []
        self.active_decay_gradients: list[torch.Tensor] = []
        self.active_no_decay_parameters: list[torch.Tensor] = []
        self.active_no_decay_gradients: list[torch.Tensor] = []
        self.momentum_buffers: dict[int, torch.Tensor] = {}
        self.accumulated_gamma = 0.0
        self._adaptive_max_lr: float | None = None

    def _compute_local_gradients(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_inputs, batch_targets = batch if batch is not None else self._next_batch()
        assert self.model is not None
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        with self._autocast_context():
            logits = self._forward_model(batch_inputs)
            loss = F.cross_entropy(
                logits.flatten(end_dim=1).float(),
                batch_targets.flatten(),
            )
        loss.backward()
        self._refresh_optimizer_gradients()
        return loss.detach().float()

    @torch.no_grad()
    def _apply_optimizer_update(self, lr: float) -> None:
        assert self.model is not None
        weight_decay = self.cfg.optimizer.weight_decay
        momentum = self.cfg.optimizer.momentum
        if momentum:
            self._apply_momentum_sgd_update(lr=lr, weight_decay=weight_decay, momentum=momentum)
            self.model.sync_storage_from_parameters_()
            return
        if self.active_decay_parameters:
            _foreach_add_(
                self.active_decay_parameters,
                self.active_decay_parameters,
                alpha=-lr * weight_decay,
            )
            _foreach_add_(
                self.active_decay_parameters,
                self.active_decay_gradients,
                alpha=-lr,
            )
        if self.active_no_decay_parameters:
            _foreach_add_(
                self.active_no_decay_parameters,
                self.active_no_decay_gradients,
                alpha=-lr,
            )
        self.model.sync_storage_from_parameters_()

    def _apply_momentum_sgd_update(
        self,
        lr: float,
        weight_decay: float,
        momentum: float,
    ) -> None:
        if self.active_decay_parameters:
            updates = self.active_decay_gradients
            if weight_decay:
                updates = _foreach_add(
                    self.active_decay_gradients,
                    self.active_decay_parameters,
                    alpha=weight_decay,
                )
            buffers = self._momentum_buffers(
                self.active_decay_parameters,
                updates,
                momentum=momentum,
            )
            _foreach_add_(self.active_decay_parameters, buffers, alpha=-lr)

        if self.active_no_decay_parameters:
            buffers = self._momentum_buffers(
                self.active_no_decay_parameters,
                self.active_no_decay_gradients,
                momentum=momentum,
            )
            _foreach_add_(self.active_no_decay_parameters, buffers, alpha=-lr)

    def _momentum_buffers(
        self,
        parameters: list[torch.Tensor],
        updates: list[torch.Tensor],
        momentum: float,
    ) -> list[torch.Tensor]:
        buffers = []
        existing_buffers = []
        existing_updates = []
        for parameter, update in zip(parameters, updates, strict=True):
            buffer = self.momentum_buffers.get(id(parameter))
            if buffer is None:
                buffer = update.detach().clone(memory_format=torch.preserve_format)
                self.momentum_buffers[id(parameter)] = buffer
            else:
                existing_buffers.append(buffer)
                existing_updates.append(update)
            buffers.append(buffer)

        if existing_buffers:
            _foreach_mul_(existing_buffers, momentum)
            _foreach_add_(existing_buffers, existing_updates, alpha=1.0)
        return buffers

    def _batches_per_epoch(self) -> int:
        if isinstance(self.dataset, PackedDataLoader):
            return len(self.dataset)
        shard_size = len(self.dataset) // self.cfg.virtual_workers
        return max(shard_size // self.cfg.data.batch_size, 1)

    def _test_batches_per_epoch(self) -> int:
        if isinstance(self.test_dataset, PackedDataLoader):
            return len(self.test_dataset)
        shard_size = len(self.test_dataset) // self.cfg.virtual_workers
        return max(shard_size // self.test_batch_size, 1)

    def _test_batch_size(self) -> int:
        if isinstance(self.test_dataset, PackedDataLoader):
            return self.test_dataset.local_batch_size
        shard_size = len(self.test_dataset) // self.cfg.virtual_workers
        if shard_size < 1:
            raise ValueError("test worker shard is empty; reduce virtual_workers")
        return min(self.cfg.data.eval_batch_size, shard_size)

    def _learning_rate(self, step: int) -> float:
        warmup_steps = self._warmup_steps()
        return self.cfg.optimizer.lr * lr_factor(
            self.cfg.scheduler,
            step,
            self.total_steps,
            warmup_steps=warmup_steps,
        )

    def _warmup_steps(self) -> int:
        if self.cfg.scheduler.name != "warmup_cosine":
            return 0
        return self.cfg.scheduler.warmup_epochs * self.batches_per_epoch

    def _should_evaluate_epoch(self, epoch: int) -> bool:
        return epoch % _EVALUATION_INTERVAL_EPOCHS == 0 or epoch == self.cfg.epochs

    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._batch_for_training_step(self.training_step)

    def _batch_for_training_step(self, training_step: int) -> tuple[torch.Tensor, torch.Tensor]:
        epoch = training_step // self.batches_per_epoch
        step = training_step % self.batches_per_epoch
        return self._batch_from_dataset(
            self.dataset,
            epoch=epoch,
            step=step,
            seed=self.cfg.data.seed,
            augment=self.cfg.data.augment,
        )

    def _batch_from_dataset(
        self,
        dataset: PackedDataLoader | InMemorySyntheticImages,
        epoch: int,
        step: int,
        seed: int,
        augment: bool,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_size or self.cfg.data.batch_size
        if isinstance(dataset, PackedDataLoader):
            if batch_size != dataset.local_batch_size:
                raise ValueError("PackedDataLoader batch size is fixed at creation")
            cached = self._epoch_batch_cache.get(id(dataset))
            if cached is None or cached[0] != epoch:
                dataset.set_epoch(epoch)
                batches = tuple(dataset)
                self._epoch_batch_cache[id(dataset)] = (epoch, batches)
            else:
                batches = cached[1]
            return batches[step % len(batches)]
        if hasattr(dataset, "batch_for_workers"):
            return dataset.batch_for_workers(
                worker_ranks=self.owned_ranks,
                virtual_workers=self.cfg.virtual_workers,
                batch_size=batch_size,
                epoch=epoch,
                step=step,
                seed=seed,
                augment=augment,
            )
        batches = [
            dataset.batch_for_worker(
                worker_rank=rank,
                virtual_workers=self.cfg.virtual_workers,
                batch_size=batch_size,
                epoch=epoch,
                step=step,
                seed=seed,
                augment=augment,
            )
            for rank in self.owned_ranks
        ]
        images = torch.stack([batch[0] for batch in batches], dim=1)
        labels = torch.stack([batch[1] for batch in batches], dim=1)
        return images, labels

    def _build_batch_prefetcher(self) -> CudaBatchPrefetcher | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        return CudaBatchPrefetcher(self)

    @torch.no_grad()
    def _evaluate_epoch(
        self,
        epoch: int,
        train_loss: float,
        lr: float,
        gamma: float = 1.0,
    ) -> EpochMetrics:
        assert self.model is not None and self.param_storage is not None
        global_vector, d2c = self._global_average_vector_and_d2c()
        was_training = self.model.training
        saved_vectors = self.param_storage.detach().clone()
        saved_buffers = self._non_parameter_storage_buffers()
        self.model.eval()
        loss_sum = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device)
        examples = torch.zeros((), device=self.device)
        try:
            self.param_storage.copy_(global_vector.expand_as(self.param_storage))
            self.model.sync_parameters_from_storage_()
            self._calibrate_average_model_batchnorm_(epoch)
            self.model.eval()
            for step in range(self.test_batches_per_epoch):
                inputs, targets = self._batch_from_dataset(
                    self.test_dataset,
                    epoch=epoch,
                    step=step,
                    seed=self.cfg.data.seed + 17_071,
                    augment=False,
                    batch_size=self.test_batch_size,
                )
                with self._autocast_context():
                    logits = self._forward_model(inputs)
                flat_logits = logits.flatten(end_dim=1)
                flat_targets = targets.flatten()
                loss_sum += F.cross_entropy(flat_logits.float(), flat_targets, reduction="sum")
                correct += flat_logits.argmax(dim=1).eq(flat_targets).sum()
                examples += flat_targets.numel()
        finally:
            self.param_storage.copy_(saved_vectors)
            self.model.sync_parameters_from_storage_()
            self._restore_buffers_(saved_buffers)
            self.model.train(was_training)

        totals = torch.stack((loss_sum, correct, examples))
        if self.ctx.is_distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        test_loss = (totals[0] / totals[2]).item()
        test_accuracy = (totals[1] / totals[2]).item()
        return EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            distance_to_consensus=d2c,
            lr=lr,
            gamma=gamma,
            accumulated_gamma=self.accumulated_gamma,
        )

    @torch.no_grad()
    def _calibrate_average_model_batchnorm_(self, epoch: int) -> None:
        assert self.model is not None
        if not self._model_has_norm_module():
            return
        self._reset_batchnorm_running_stats_()
        self.model.train(True)
        calibration_epoch = max(epoch - 1, 0)
        for step in range(self.batches_per_epoch):
            inputs, _ = self._batch_from_dataset(
                self.dataset,
                epoch=calibration_epoch,
                step=step,
                seed=self.cfg.data.seed,
                augment=self.cfg.data.augment,
            )
            with self._autocast_context():
                self._forward_model(inputs)
        self._average_packed_buffers_()

    @torch.no_grad()
    def _reset_batchnorm_running_stats_(self) -> None:
        assert self.model is not None
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()

    def _local_vectors(self) -> torch.Tensor:
        assert self.param_storage is not None
        return self.param_storage

    def _load_local_vectors_(self, vectors: torch.Tensor) -> None:
        assert self.model is not None and self.param_storage is not None
        self.param_storage.copy_(vectors)
        self.model.sync_parameters_from_storage_()

    def _gather_global_vectors(self) -> torch.Tensor:
        return all_gather_owned_buckets(self._local_vectors(), self.ctx)

    def _global_average_vector_and_d2c(self) -> tuple[torch.Tensor, float]:
        vectors = self._gather_global_vectors()
        d2c = average_distance_to_consensus(vectors)
        global_vector = vectors.mean(dim=0)
        return global_vector, d2c

    def _autocast_context(self):
        if not self._amp_enabled():
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            cache_enabled=False,
        )

    def _amp_enabled(self) -> bool:
        if not self.cfg.runtime.amp or self.device.type not in {"cpu", "cuda"}:
            return False
        if self.device.type == "cpu" and self._model_has_norm_module():
            return False
        return True

    def _model_has_norm_module(self) -> bool:
        assert self.model is not None
        return any(isinstance(module, NORM_MODULES) for module in self.model.modules())

    def _init_data(self, train: bool) -> PackedDataLoader | InMemorySyntheticImages:
        if self.cfg.data.dataset == DatasetName.SYNTHETIC:
            return InMemorySyntheticImages(
                samples=self.cfg.virtual_workers * self.cfg.data.batch_size * 4,
                num_classes=self.cfg.data.num_classes,
                seed=self.cfg.data.seed if train else self.cfg.data.seed + 1,
                device=self.device,
            )
        local_batch_size = self.cfg.data.batch_size if train else self.cfg.data.eval_batch_size
        loader = create_dataloader(
            self.cfg.data.dataset.value.lower(),
            root=self.cfg.data.root,
            local_batch_size=local_batch_size,
            world_size=self.cfg.virtual_workers,
            ranks=self.owned_ranks,
            base_seed=self.cfg.data.seed if train else self.cfg.data.seed + 17_071,
            train=train,
            packed=True,
            channels_last=True,
            shuffle=train,
            augment=self.cfg.data.augment if train else False,
            device=self.device,
            sampler_drop_last=train,
            drop_last=train,
        )
        if len(loader) == 0:
            raise ValueError(
                "CIFAR worker shard has no complete batches; reduce virtual_workers or batch_size"
            )
        return loader

    def _init_packed_model(self) -> None:
        torch.manual_seed(self.cfg.seed)
        num_classes = num_classes_for_dataset(self.cfg.data.dataset, self.cfg.data.num_classes)
        self.model = cast(
            PackedModel,
            get_packed_model(
                self.cfg.model.name,
                num_classes=num_classes,
                num_models=self.local_worker_count,
            ).to(self.device),
        )
        self.model.train()
        self._model_forward = self._build_model_forward()
        param_storage = self.model.parameter_storage
        self.param_storage = param_storage
        self.param_layout = parameter_storage_layout(
            cast(nn.Module, self.model),
            self.local_worker_count,
        )
        self._synchronize_initial_replicas_()
        self._init_optimizer_parameters(
            parameter_decay_mask(cast(nn.Module, self.model), self.cfg.optimizer.weight_decay)
        )
        logger.info(
            "Rank {} initialized packed model with {} local replicas and {} parameter tensors "
            "coalesced into {} scalars per worker",
            self.ctx.rank,
            self.local_worker_count,
            len(tuple(self.model.named_parameters())),
            param_storage.size(1),
        )

    @torch.no_grad()
    def _synchronize_initial_replicas_(self) -> None:
        assert self.model is not None and self.param_storage is not None
        source = self.param_storage[0].detach().clone()
        if self.ctx.is_distributed:
            dist.broadcast(source, src=0)
        self.param_storage.copy_(source.expand_as(self.param_storage))
        self.model.sync_parameters_from_storage_()

    def _build_model_forward(self) -> Callable[[torch.Tensor], torch.Tensor]:
        assert self.model is not None

        def forward(inputs: torch.Tensor) -> torch.Tensor:
            assert self.model is not None
            return self.model(inputs)

        if not self.cfg.runtime.compile:
            return forward
        compiled = torch.compile(
            forward,
            mode=self.cfg.runtime.compile_mode,
            fullgraph=False,
        )
        logger.info(
            "Rank {} enabled torch.compile for model forward with mode={}",
            self.ctx.rank,
            self.cfg.runtime.compile_mode,
        )
        return cast(Callable[[torch.Tensor], torch.Tensor], compiled)

    def _forward_model(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self._model_forward is not None
        return self._model_forward(inputs)

    def _init_optimizer_parameters(self, decay_by_name: dict[str, float]) -> None:
        assert self.model is not None
        self.decay_parameters = []
        self.no_decay_parameters = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if decay_by_name.get(name, 0.0):
                self.decay_parameters.append(parameter)
            else:
                self.no_decay_parameters.append(parameter)

    def _refresh_optimizer_gradients(self) -> None:
        self.active_decay_parameters, self.active_decay_gradients = _parameters_with_gradients(
            self.decay_parameters
        )
        self.active_no_decay_parameters, self.active_no_decay_gradients = (
            _parameters_with_gradients(self.no_decay_parameters)
        )

    def _non_parameter_storage_buffers(self) -> dict[str, torch.Tensor]:
        assert self.model is not None
        return {
            name: buffer.detach().clone()
            for name, buffer in self.model.named_buffers()
            if name != "parameter_storage"
        }

    def _restore_buffers_(self, buffers: dict[str, torch.Tensor]) -> None:
        assert self.model is not None
        current = dict(self.model.named_buffers())
        for name, value in buffers.items():
            current[name].copy_(value)

    @torch.no_grad()
    def _average_packed_buffers_(self) -> None:
        assert self.model is not None
        for module in self.model.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            running_mean = module.running_mean
            running_var = module.running_var
            if running_mean is None or running_var is None:
                continue
            num_models = getattr(module, "num_models", 1)
            local_features = getattr(module, "local_num_features", module.num_features)
            for buffer in (running_mean, running_var):
                value = buffer.view(num_models, local_features).mean(dim=0)
                if self.ctx.is_distributed:
                    dist.all_reduce(value, op=dist.ReduceOp.SUM)
                    value.div_(self.ctx.world_size)
                buffer.copy_(value.repeat(num_models))
            if module.num_batches_tracked is not None and self.ctx.is_distributed:
                dist.all_reduce(module.num_batches_tracked, op=dist.ReduceOp.MAX)

    def _use_cuda_amp_batched_autograd(self) -> bool:
        return False

    def _packed_storage_value(self, name: str) -> torch.Tensor:
        assert self.model is not None and self.param_storage is not None
        for item in self.param_layout:
            if item.name == name or item.name.endswith(f".{name}"):
                return self.param_storage[:, item.start : item.start + item.numel].view(
                    self.local_worker_count,
                    *item.shape,
                )
        raise KeyError(name)

    def _owned_ranks(self) -> tuple[int, ...]:
        workers_per_process = self.cfg.virtual_workers // self.ctx.world_size
        start = self.ctx.rank * workers_per_process
        return tuple(range(start, start + workers_per_process))

    def _process_for_worker(self, worker_rank: int) -> int:
        return worker_rank // self.workers_per_process

    def _validate_process_layout(self) -> None:
        if self.ctx.world_size < 1 or self.ctx.world_size & (self.ctx.world_size - 1):
            raise ValueError("process world_size must be a positive power of two")
        if self.cfg.virtual_workers % self.ctx.world_size != 0:
            raise ValueError(
                "virtual_workers must be divisible by launched process count; "
                f"got {self.cfg.virtual_workers} workers and {self.ctx.world_size} processes"
            )


def _parameters_with_gradients(
    parameters: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    active_parameters = []
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        active_parameters.append(parameter)
        gradients.append(parameter.grad)
    return active_parameters, gradients


def _foreach_add_(
    tensors: list[torch.Tensor],
    others: list[torch.Tensor],
    alpha: float,
) -> None:
    foreach_add = cast(Any, getattr(torch, _FOREACH_ADD_INPLACE))
    foreach_add(tensors, others, alpha=alpha)


def _foreach_add(
    tensors: list[torch.Tensor],
    others: list[torch.Tensor],
    alpha: float,
) -> list[torch.Tensor]:
    foreach_add = cast(Any, getattr(torch, _FOREACH_ADD))
    return foreach_add(tensors, others, alpha=alpha)


def _foreach_mul_(tensors: list[torch.Tensor], scalar: float) -> None:
    foreach_mul = cast(Any, getattr(torch, _FOREACH_MUL_INPLACE))
    foreach_mul(tensors, scalar)
