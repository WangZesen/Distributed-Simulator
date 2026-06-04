from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from loguru import logger

from distributed_simulator.config import (
    AdaptiveMixConfig,
    DecentralizedTrainerConfig,
    SimulationConfig,
    Topology,
)
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.topology import communication_schedule, groups_for_rank
from distributed_simulator.trainers.base import BaseTrainer, TrainMetrics


@dataclass
class PendingPairwiseExchange:
    recv_by_process: dict[int, tuple[int, ...]]
    recv_tensors: dict[int, torch.Tensor]
    works: list[dist.Work]
    send_tensors: list[torch.Tensor]


@dataclass
class PairwiseExchangePlan:
    peer_by_rank: dict[int, int]
    peer_local_indices: torch.Tensor
    send_local_indices_by_process: dict[int, torch.Tensor]
    recv_by_process: dict[int, tuple[int, ...]]
    remote_processes: tuple[int, ...]


@dataclass
class PendingMix:
    stream: torch.cuda.Stream | None
    vectors: torch.Tensor
    local_vectors: torch.Tensor
    gamma: float
    mixed: torch.Tensor | None = None
    all_reduce_work: dist.Work | None = None
    pairwise_peer_by_rank: dict[int, int] | None = None
    pairwise_plan: PairwiseExchangePlan | None = None
    pairwise_exchange: PendingPairwiseExchange | None = None


def _peer_for_rank(rank: int, ranks: tuple[int, ...]) -> int:
    if len(ranks) == 1:
        return rank
    if len(ranks) == 2:
        first, second = ranks
        return second if rank == first else first
    raise ValueError("pairwise topology mix expected groups of size one or two")


class DecentralizedTrainer(BaseTrainer):
    """Standard decentralized training over simulated virtual workers.

    Local virtual workers are stored in one packed module. Each iteration
    computes isolated local gradients with one regular autograd pass, mixes
    packed parameter storage with the active decentralized topology, and then
    applies the optimizer update to the mixed parameters.
    """

    def __init__(self, cfg: SimulationConfig, ctx: ProcessContext | None = None):
        trainer_cfg = cfg.trainer
        if not isinstance(trainer_cfg, DecentralizedTrainerConfig):
            raise ValueError("DecentralizedTrainer requires a decentralized trainer config")
        super().__init__(cfg, trainer_cfg, ctx)

        self.schedule = communication_schedule(self.trainer_cfg.topology, cfg.virtual_workers)
        self.groups_by_rank = {
            rank: groups_for_rank(self.schedule, rank) for rank in self.owned_ranks
        }
        self.pairwise_exchange_plans = self._build_pairwise_exchange_plans()
        self._init_packed_model()
        logger.info(
            "Rank {} runtime: amp={} dtype={} compile={} compile_mode={} "
            "overlap_mixing={} backend=packed",
            self.ctx.rank,
            self._amp_enabled(),
            self.cfg.runtime.amp_dtype,
            self.cfg.runtime.compile,
            self.cfg.runtime.compile_mode,
            self.trainer_cfg.overlap_mixing,
        )
        logger.debug(
            "Rank {} owns virtual workers {} on {}",
            self.ctx.rank,
            self.owned_ranks,
            self.device,
        )

    def train(self) -> TrainMetrics:
        logger.info(
            "Rank {} starting decentralized training for {} epochs "
            "({} steps) with {} local workers",
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
            gamma = self._mixing_gamma(step, current_lr)
            batch = None
            if prefetcher is not None and prefetched_batch is not None:
                batch = prefetcher.wait(prefetched_batch)
                next_step = step + 1
                prefetched_batch = (
                    prefetcher.prefetch(next_step) if next_step < self.total_steps else None
                )
            if self._use_cuda_mixing_overlap():
                pending_mix = self._start_mixing(step, gamma=gamma)
                loss = self._compute_local_gradients(batch=batch)
                self._finish_mixing(pending_mix)
            else:
                loss = self._compute_local_gradients(batch=batch)
                self._mix_parameters(step, gamma=gamma)
            self._apply_optimizer_update(current_lr)
            self.accumulated_gamma += gamma
            loss_value = loss.detach().float()
            total_loss_sum.add_(loss_value)
            epoch_loss_sum.add_(loss_value)
            completed_steps += 1
            if (step + 1) % self.batches_per_epoch == 0:
                epoch = (step + 1) // self.batches_per_epoch
                train_loss = (epoch_loss_sum / self.batches_per_epoch).item()
                epoch_loss_sum.zero_()
                if self._should_evaluate_epoch(epoch):
                    metrics = self._evaluate_epoch(epoch, train_loss, current_lr, gamma)
                    history.append(metrics)
                    if self.ctx.rank == 0:
                        logger.info(
                            "epoch={} train_loss={:.6f} test_loss={:.6f} "
                            "test_acc={:.4f} d2c={:.6f} lr={:.6g} "
                            "gamma={:.6g} accum_gamma={:.6g}",
                            metrics.epoch,
                            metrics.train_loss,
                            metrics.test_loss,
                            metrics.test_accuracy,
                            metrics.distance_to_consensus,
                            metrics.lr,
                            metrics.gamma,
                            metrics.accumulated_gamma,
                        )

        final_metrics = (
            history[-1]
            if history
            else self._evaluate_epoch(0, 0.0, 0.0, self._mixing_gamma(0, 0.0))
        )
        local_loss = (total_loss_sum / completed_steps).item() if completed_steps else 0.0
        logger.info(
            "Rank {} finished decentralized training: loss={:.6f} d2c={:.6f}",
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
            gamma=final_metrics.gamma,
            accumulated_gamma=final_metrics.accumulated_gamma,
            epochs=self.cfg.epochs,
            steps=self.total_steps,
            rank=self.ctx.rank,
            world_size=self.ctx.world_size,
            owned_workers=self.owned_ranks,
            history=tuple(history),
        )

    @torch.no_grad()
    def _mix_parameters(self, step: int, gamma: float | None = None) -> None:
        gamma = self._mixing_gamma(step, self._learning_rate(step)) if gamma is None else gamma
        self._finish_mixing(self._start_mixing(step, gamma=gamma, overlap=False))

    @torch.no_grad()
    def _start_mixing(
        self,
        step: int,
        gamma: float | None = None,
        overlap: bool | None = None,
    ) -> PendingMix:
        assert self.model is not None
        gamma = self._mixing_gamma(step, self._learning_rate(step)) if gamma is None else gamma
        use_overlap = self._use_cuda_mixing_overlap() if overlap is None else overlap
        if not use_overlap:
            return self._start_mixing_on_current_stream(step, gamma=gamma, stream=None)

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            return self._start_mixing_on_current_stream(step, gamma=gamma, stream=stream)

    @torch.no_grad()
    def _start_mixing_on_current_stream(
        self,
        step: int,
        gamma: float,
        stream: torch.cuda.Stream | None,
    ) -> PendingMix:
        assert self.model is not None
        self.model.sync_storage_from_parameters_()
        vectors = self._local_vectors()
        if self.trainer_cfg.topology == Topology.COMPLETE:
            return self._start_complete_graph_mix(vectors, gamma=gamma, stream=stream)
        return self._start_pairwise_topology_mix(vectors, step=step, gamma=gamma, stream=stream)

    @torch.no_grad()
    def _finish_mixing(self, pending: PendingMix) -> None:
        if pending.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(pending.stream)

        if self.trainer_cfg.topology == Topology.COMPLETE:
            self._finish_complete_graph_mix_into_storage_(pending)
            return

        mixed = self._finish_mixed_vectors(pending)
        self._load_local_vectors_(mixed)

    @torch.no_grad()
    def _finish_mixed_vectors(self, pending: PendingMix) -> torch.Tensor:
        if (
            pending.pairwise_exchange is not None
            and pending.pairwise_peer_by_rank is not None
            and pending.pairwise_plan is not None
        ):
            remote_peer_vectors = self._finish_remote_peer_exchange(pending.pairwise_exchange)
            mixed = self._finish_pairwise_topology_mix(
                pending.vectors,
                pending.pairwise_peer_by_rank,
                remote_peer_vectors,
                peer_local_indices=pending.pairwise_plan.peer_local_indices,
            )
            return self._apply_mix_gamma(pending.local_vectors, mixed, pending.gamma)
        assert pending.mixed is not None
        return self._apply_mix_gamma(pending.local_vectors, pending.mixed, pending.gamma)

    @torch.no_grad()
    def _start_complete_graph_mix(
        self,
        local_vectors: torch.Tensor,
        gamma: float,
        stream: torch.cuda.Stream | None,
    ) -> PendingMix:
        local_mean = local_vectors.mean(dim=0)
        if not self.ctx.is_distributed:
            return PendingMix(
                stream=stream,
                vectors=local_mean,
                local_vectors=local_vectors,
                gamma=gamma,
            )

        logger.debug("Rank {} mixing complete topology with one all-reduce", self.ctx.rank)
        work = dist.all_reduce(local_mean, op=dist.ReduceOp.SUM, async_op=True)
        return PendingMix(
            stream=stream,
            vectors=local_mean,
            local_vectors=local_vectors,
            gamma=gamma,
            all_reduce_work=work,
        )

    @torch.no_grad()
    def _finish_complete_graph_mix_into_storage_(self, pending: PendingMix) -> None:
        assert self.model is not None
        if pending.all_reduce_work is not None:
            pending.all_reduce_work.wait()
            pending.vectors.div_(self.ctx.world_size)

        mean = pending.vectors.expand_as(pending.local_vectors)
        if pending.gamma == 1.0:
            pending.local_vectors.copy_(mean)
        else:
            pending.local_vectors.lerp_(mean, pending.gamma)
        self.model.sync_parameters_from_storage_()

    @torch.no_grad()
    def _start_pairwise_topology_mix(
        self,
        local_vectors: torch.Tensor,
        step: int,
        gamma: float,
        stream: torch.cuda.Stream | None,
    ) -> PendingMix:
        plan = self._pairwise_exchange_plan(step)
        exchange = self._start_remote_peer_exchange(local_vectors, plan)
        if exchange is None:
            mixed = self._finish_pairwise_topology_mix(
                local_vectors,
                plan.peer_by_rank,
                {},
                peer_local_indices=plan.peer_local_indices,
            )
            return PendingMix(
                stream=stream,
                vectors=local_vectors,
                local_vectors=local_vectors,
                gamma=gamma,
                mixed=mixed,
            )
        return PendingMix(
            stream=stream,
            vectors=local_vectors,
            local_vectors=local_vectors,
            gamma=gamma,
            pairwise_peer_by_rank=plan.peer_by_rank,
            pairwise_plan=plan,
            pairwise_exchange=exchange,
        )

    @torch.no_grad()
    def _finish_pairwise_topology_mix(
        self,
        local_vectors: torch.Tensor,
        peer_by_rank: dict[int, int],
        remote_peer_vectors: dict[int, torch.Tensor],
        peer_local_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if peer_local_indices is not None and not remote_peer_vectors:
            peer_vectors = local_vectors.index_select(0, peer_local_indices)
            return torch.add(local_vectors, peer_vectors).mul_(0.5)

        peer_vectors = torch.empty_like(local_vectors)

        for local_index, rank in enumerate(self.owned_ranks):
            peer = peer_by_rank[rank]
            if peer == rank:
                peer_vectors[local_index].copy_(local_vectors[local_index])
            elif peer in self.local_index_by_rank:
                peer_vectors[local_index].copy_(local_vectors[self.local_index_by_rank[peer]])
            else:
                peer_vectors[local_index].copy_(remote_peer_vectors[peer])
        return torch.add(local_vectors, peer_vectors).mul_(0.5)

    def _mixing_gamma(self, step: int, lr: float) -> float:
        mix_cfg = self.trainer_cfg.mix
        if mix_cfg.name == "normal":
            return 1.0

        if not isinstance(mix_cfg, AdaptiveMixConfig):
            raise ValueError(f"unsupported decentralized mix config: {mix_cfg.name}")

        if mix_cfg.p < 0:
            return mix_cfg.min_gamma

        epoch = step // self.batches_per_epoch if self.batches_per_epoch else 0
        if epoch < mix_cfg.start_epoch:
            return mix_cfg.max_gamma

        if self._adaptive_max_lr is None:
            self._adaptive_max_lr = lr

        if self._adaptive_max_lr <= 0.0:
            lr_ratio = 0.0
        else:
            lr_ratio = lr / self._adaptive_max_lr
        gamma = (lr_ratio**mix_cfg.p) * (mix_cfg.max_gamma - mix_cfg.min_gamma)
        return gamma + mix_cfg.min_gamma

    def _apply_mix_gamma(
        self,
        local_vectors: torch.Tensor,
        mixed_vectors: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        if gamma == 1.0:
            return mixed_vectors
        return local_vectors.lerp(mixed_vectors, gamma)

    @torch.no_grad()
    def _start_remote_peer_exchange(
        self,
        local_vectors: torch.Tensor,
        plan: PairwiseExchangePlan,
    ) -> PendingPairwiseExchange | None:
        if not self.ctx.is_distributed or not plan.remote_processes:
            return None

        ops: list[dist.P2POp] = []
        recv_tensors: dict[int, torch.Tensor] = {}
        send_tensors: list[torch.Tensor] = []
        for process in plan.remote_processes:
            send_indices = plan.send_local_indices_by_process.get(process)
            recv_ranks = plan.recv_by_process.get(process, ())
            if send_indices is not None:
                send_tensor = local_vectors.index_select(0, send_indices)
                send_tensors.append(send_tensor)
                ops.append(dist.P2POp(dist.isend, send_tensor, process))
            if recv_ranks:
                recv_tensor = torch.empty(
                    len(recv_ranks),
                    local_vectors.size(1),
                    dtype=local_vectors.dtype,
                    device=local_vectors.device,
                )
                recv_tensors[process] = recv_tensor
                ops.append(dist.P2POp(dist.irecv, recv_tensor, process))

        if ops:
            logger.debug(
                "Rank {} exchanging pairwise parameters with {} remote process(es)",
                self.ctx.rank,
                len(plan.remote_processes),
            )
            works = list(dist.batch_isend_irecv(ops))
            return PendingPairwiseExchange(
                recv_by_process=plan.recv_by_process,
                recv_tensors=recv_tensors,
                works=works,
                send_tensors=send_tensors,
            )
        return None

    @torch.no_grad()
    def _finish_remote_peer_exchange(
        self,
        exchange: PendingPairwiseExchange,
    ) -> dict[int, torch.Tensor]:
        for work in exchange.works:
            work.wait()
        remote_vectors: dict[int, torch.Tensor] = {}
        for process, tensor in exchange.recv_tensors.items():
            for rank, vector in zip(sorted(exchange.recv_by_process[process]), tensor, strict=True):
                remote_vectors[rank] = vector
        return remote_vectors

    def _active_peer_by_rank(self, step: int) -> dict[int, int]:
        if not self.pairwise_exchange_plans:
            return {rank: rank for rank in self.owned_ranks}
        return dict(self._pairwise_exchange_plan(step).peer_by_rank)

    def _pairwise_exchange_plan(self, step: int) -> PairwiseExchangePlan:
        if not self.pairwise_exchange_plans:
            raise ValueError("pairwise exchange plan requested for complete topology")
        return self.pairwise_exchange_plans[step % len(self.pairwise_exchange_plans)]

    def _build_pairwise_exchange_plans(self) -> tuple[PairwiseExchangePlan, ...]:
        if self.trainer_cfg.topology == Topology.COMPLETE:
            return ()
        if not self.owned_ranks:
            return ()

        phase_count = max(len(groups) for groups in self.groups_by_rank.values())
        plans = []
        for phase in range(phase_count):
            peer_by_rank: dict[int, int] = {}
            peer_local_indices = []
            send_by_process: dict[int, list[tuple[int, int]]] = {}
            recv_by_process: dict[int, list[int]] = {}

            for local_index, rank in enumerate(self.owned_ranks):
                local_groups = self.groups_by_rank[rank]
                active = local_groups[phase % len(local_groups)]
                peer = _peer_for_rank(rank, active.ranks)
                peer_by_rank[rank] = peer
                peer_local_index = self.local_index_by_rank.get(peer, -1)
                peer_local_indices.append(peer_local_index)

                peer_process = self._process_for_worker(peer)
                if peer_process != self.ctx.rank:
                    send_by_process.setdefault(peer_process, []).append((rank, local_index))
                    recv_by_process.setdefault(peer_process, []).append(peer)

            send_indices = {
                process: torch.tensor(
                    [local_index for _, local_index in sorted(items)],
                    dtype=torch.long,
                    device=self.device,
                )
                for process, items in send_by_process.items()
            }
            recv_ranks = {
                process: tuple(sorted(ranks)) for process, ranks in recv_by_process.items()
            }
            remote_processes = tuple(sorted(set(send_indices) | set(recv_ranks)))
            plans.append(
                PairwiseExchangePlan(
                    peer_by_rank=peer_by_rank,
                    peer_local_indices=torch.tensor(
                        peer_local_indices,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    send_local_indices_by_process=send_indices,
                    recv_by_process=recv_ranks,
                    remote_processes=remote_processes,
                )
            )
        return tuple(plans)

    def _use_cuda_mixing_overlap(self) -> bool:
        return (
            self.trainer_cfg.overlap_mixing
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
