from __future__ import annotations

import torch
import torch.distributed as dist
from loguru import logger

from distributed_simulator.config import SAMTrainerConfig, SimulationConfig
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.model import packed_parameter_view
from distributed_simulator.trainers.base import BaseTrainer, TrainMetrics


class SAMTrainer(BaseTrainer):
    """Sharpness-Aware Minimization over synchronous virtual workers."""

    def __init__(self, cfg: SimulationConfig, ctx: ProcessContext | None = None):
        trainer_cfg = cfg.trainer
        if not isinstance(trainer_cfg, SAMTrainerConfig):
            raise ValueError("SAMTrainer requires a SAM trainer config")
        super().__init__(cfg, trainer_cfg, ctx)

        self._init_packed_model()
        assert self.model is not None and self.param_storage is not None
        self._parameter_by_storage_name = dict(self.model.named_parameters())
        self._gradient_storage_buffer = torch.empty_like(self.param_storage)
        self._perturbation_buffer = torch.empty_like(self.param_storage)
        self._base_parameter_buffer = torch.empty_like(self.param_storage)
        self._foreach_parameters = []
        self._foreach_perturbations = []
        self._foreach_base_parameters = []
        for item in self.param_layout:
            parameter = self._parameter_by_storage_name[item.name]
            if not parameter.requires_grad:
                continue
            self._foreach_parameters.append(parameter)
            self._foreach_perturbations.append(
                self._perturbation_buffer[:, item.start : item.start + item.numel]
            )
            self._foreach_base_parameters.append(
                self._base_parameter_buffer[:, item.start : item.start + item.numel]
            )
        self._sam_scale_buffer = torch.empty(
            self.local_worker_count,
            dtype=self.param_storage.dtype,
            device=self.param_storage.device,
        )
        self._averaged_gradient_buffer = torch.empty(
            self.param_storage.size(1),
            dtype=self.param_storage.dtype,
            device=self.param_storage.device,
        )
        logger.info(
            "Rank {} runtime: amp={} dtype={} tf32={} compile={} compile_mode={} "
            "backend=packed-sam rho={}",
            self.ctx.rank,
            self._amp_enabled(),
            self.cfg.runtime.amp_dtype,
            self.tf32_enabled,
            self.cfg.runtime.compile,
            self.cfg.runtime.compile_mode,
            trainer_cfg.rho,
        )
        logger.debug(
            "Rank {} owns virtual workers {} on {}",
            self.ctx.rank,
            self.owned_ranks,
            self.device,
        )

    def train(self) -> TrainMetrics:
        logger.info(
            "Rank {} starting SAM training for {} epochs ({} steps) with {} local workers",
            self.ctx.rank,
            self.cfg.epochs,
            self.total_steps,
            self.local_worker_count,
        )
        total_loss_sum = torch.zeros((), device=self.device)
        epoch_loss_sum = torch.zeros((), device=self.device)
        completed_steps = 0
        history = []
        prefetcher = self._build_batch_prefetcher()
        prefetched_batch = (
            prefetcher.prefetch(0) if prefetcher is not None and self.total_steps else None
        )
        for step in range(self.total_steps):
            self.training_step = step
            current_lr = self._learning_rate(step)
            batch = None
            if prefetcher is not None and prefetched_batch is not None:
                batch = prefetcher.wait(prefetched_batch)
                next_step = step + 1
                prefetched_batch = (
                    prefetcher.prefetch(next_step) if next_step < self.total_steps else None
                )
            loss = self._compute_sam_gradients(batch=batch)
            self._average_gradients_()
            self._apply_optimizer_update(current_lr)
            loss_value = loss.detach().float()
            total_loss_sum.add_(loss_value)
            epoch_loss_sum.add_(loss_value)
            completed_steps += 1
            if (step + 1) % self.batches_per_epoch == 0:
                epoch = (step + 1) // self.batches_per_epoch
                train_loss_sum = epoch_loss_sum.detach().clone()
                if self.ctx.is_distributed:
                    dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
                    train_loss_sum.div_(self.ctx.world_size)
                train_loss = (train_loss_sum / self.batches_per_epoch).item()
                epoch_loss_sum.zero_()
                if self._should_evaluate_epoch(epoch):
                    metrics = self._evaluate_epoch(epoch, train_loss, current_lr, gamma=0.0)
                    history.append(metrics)
                    if self.ctx.rank == 0:
                        logger.info(
                            "epoch={} train_loss={:.6f} test_loss={:.6f} "
                            "test_acc={:.4f} d2c={:.6f} lr={:.6g}",
                            metrics.epoch,
                            metrics.train_loss,
                            metrics.test_loss,
                            metrics.test_accuracy,
                            metrics.distance_to_consensus,
                            metrics.lr,
                        )

        final_metrics = history[-1] if history else self._evaluate_epoch(0, 0.0, 0.0, gamma=0.0)
        loss_sum = total_loss_sum.detach().clone()
        if self.ctx.is_distributed:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            loss_sum.div_(self.ctx.world_size)
        local_loss = (loss_sum / completed_steps).item() if completed_steps else 0.0
        logger.info(
            "Rank {} finished SAM training: loss={:.6f} d2c={:.6f}",
            self.ctx.rank,
            local_loss,
            final_metrics.distance_to_consensus,
        )
        return TrainMetrics(
            loss=local_loss,
            distance_to_consensus=final_metrics.distance_to_consensus,
            test_loss=final_metrics.test_loss,
            test_accuracy=final_metrics.test_accuracy,
            lr=final_metrics.lr,
            gamma=0.0,
            accumulated_gamma=0.0,
            epochs=self.cfg.epochs,
            steps=self.total_steps,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            owned_workers=self.owned_ranks,
            history=tuple(history),
        )

    def _compute_sam_gradients(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        assert self.model is not None and self.param_storage is not None
        batch_inputs, batch_targets = batch if batch is not None else self._next_batch()

        self._compute_local_gradients((batch_inputs, batch_targets))
        self._pack_current_gradients_(self._gradient_storage_buffer)
        self._base_parameter_buffer.copy_(self.param_storage)
        self._perturb_parameters_()

        second_loss = self._compute_local_gradients((batch_inputs, batch_targets))
        with torch.no_grad():
            torch._foreach_copy_(
                [
                    packed_parameter_view(parameter, self.local_worker_count)
                    for parameter in self._foreach_parameters
                ],
                self._foreach_base_parameters,
            )
            self.model.sync_storage_from_parameters_()
        return second_loss

    @torch.no_grad()
    def _perturb_parameters_(self) -> None:
        assert self.model is not None and self.param_storage is not None
        trainer_cfg = self.trainer_cfg
        assert isinstance(trainer_cfg, SAMTrainerConfig)
        norms = torch.linalg.vector_norm(self._gradient_storage_buffer.float(), ord=2, dim=1)
        torch.reciprocal(norms.add_(1e-9), out=self._sam_scale_buffer)
        self._sam_scale_buffer.mul_(trainer_cfg.rho)
        self._perturbation_buffer.copy_(self._gradient_storage_buffer)
        self._perturbation_buffer.mul_(self._sam_scale_buffer.unsqueeze(1))
        torch._foreach_add_(
            [
                packed_parameter_view(parameter, self.local_worker_count)
                for parameter in self._foreach_parameters
            ],
            self._foreach_perturbations,
        )
        self.model.sync_storage_from_parameters_()
        self.model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _average_gradients_(self) -> None:
        assert self.model is not None
        averaged = self._coalesced_averaged_gradient_()
        for item in self.param_layout:
            parameter = self._parameter_by_storage_name[item.name]
            if parameter.grad is None:
                continue
            segment = averaged[item.start : item.start + item.numel]
            grad_by_worker = packed_parameter_view(parameter.grad, self.local_worker_count)
            grad_by_worker.copy_(segment.expand_as(grad_by_worker))

    def _coalesced_averaged_gradient_(self) -> torch.Tensor:
        self._pack_current_gradients_(self._gradient_storage_buffer)
        averaged = self._averaged_gradient_buffer
        torch.mean(self._gradient_storage_buffer, dim=0, out=averaged)
        if self.ctx.is_distributed:
            dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
            averaged.div_(self.ctx.world_size)
        return averaged

    @torch.no_grad()
    def _pack_current_gradients_(self, gradient_storage: torch.Tensor) -> None:
        assert self.model is not None
        gradient_storage.zero_()
        for item in self.param_layout:
            parameter = self._parameter_by_storage_name[item.name]
            if parameter.grad is None:
                continue
            gradient_storage[:, item.start : item.start + item.numel].copy_(
                packed_parameter_view(parameter.grad.detach(), self.local_worker_count)
            )
