import subprocess
import sys

import pytest
import torch

import distributed_simulator.trainer as trainer_module
from distributed_simulator.config import (
    AdaptiveMixConfig,
    DataConfig,
    DecentralizedConfig,
    DecentralizedTrainerConfig,
    ModelConfig,
    NormalMixConfig,
    OptimizerConfig,
    RuntimeConfig,
    Topology,
    WarmupCosineSchedulerConfig,
)
from distributed_simulator.data import DatasetName, deterministic_worker_indices
from distributed_simulator.distributed import ProcessContext, resolve_process_device
from distributed_simulator.model import ModelName
from distributed_simulator.trainer import DecentralizedTrainer


def test_standard_decentralized_cpu_smoke_single_process() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=4,
        trainer=DecentralizedTrainerConfig(topology=Topology.RING),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    metrics = trainer.train()

    assert metrics.epochs == 1
    assert metrics.steps == 4
    assert metrics.owned_workers == (0, 1, 2, 3)
    assert len(metrics.history) == 1
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert torch.isfinite(torch.tensor(metrics.test_loss))
    assert 0.0 <= metrics.test_accuracy <= 1.0
    assert metrics.distance_to_consensus >= 0.0
    assert metrics.history[0].train_loss == metrics.loss
    assert metrics.history[0].test_loss == metrics.test_loss
    assert metrics.history[0].test_accuracy == metrics.test_accuracy


def test_synthetic_batches_are_image_shaped() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=3, num_classes=4),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    images, labels = trainer._next_batch()

    assert images.shape == (3, 2, 3, 32, 32)
    assert labels.shape == (3, 2)


def test_training_loop_consumes_prefetched_batches(monkeypatch) -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    prefetched_steps = []
    waited_steps = []

    class FakePrefetcher:
        def prefetch(self, step: int):  # noqa: ANN202
            prefetched_steps.append(step)
            return step

        def wait(self, batch):  # noqa: ANN001, ANN202
            waited_steps.append(batch)
            return trainer._batch_for_training_step(batch)

    monkeypatch.setattr(trainer, "_build_batch_prefetcher", lambda: FakePrefetcher())

    metrics = trainer.train()

    assert metrics.steps == trainer.total_steps
    assert prefetched_steps == list(range(trainer.total_steps))
    assert waited_steps == list(range(trainer.total_steps))


def test_decentralized_trainer_uses_packed_storage() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    assert trainer.param_storage is not None
    assert trainer._local_vectors() is trainer.param_storage
    assert trainer.model is not None
    assert hasattr(trainer.model, "parameter_storage_layout")


def test_resolve_process_device_maps_cuda_to_local_rank(monkeypatch) -> None:
    selected_devices = []
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)

    device = resolve_process_device("cuda")

    assert device == torch.device("cuda:1")
    assert selected_devices == [torch.device("cuda:1")]


def test_resolve_process_device_keeps_explicit_cuda_index(monkeypatch) -> None:
    selected_devices = []
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)

    device = resolve_process_device("cuda:0")

    assert device == torch.device("cuda:0")
    assert selected_devices == [torch.device("cuda:0")]


def test_resolve_process_device_leaves_single_process_cuda_unindexed(monkeypatch) -> None:
    selected_devices = []
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)

    device = resolve_process_device("cuda")

    assert device == torch.device("cuda")
    assert selected_devices == []


def test_complete_mix_happens_before_optimizer_update() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=4, num_classes=2),
    )
    actual = DecentralizedTrainer(cfg, ProcessContext())
    expected = DecentralizedTrainer(cfg, ProcessContext())

    with torch.no_grad():
        for trainer in (actual, expected):
            trainer._packed_storage_value("weight")[0].fill_(0.25)
            trainer._packed_storage_value("weight")[1].fill_(-0.75)
            trainer._packed_storage_value("bias")[0].fill_(0.10)
            trainer._packed_storage_value("bias")[1].fill_(-0.20)
            assert trainer.model is not None
            trainer.model.sync_parameters_from_storage_()

    for step in range(expected.total_steps):
        expected.training_step = step
        expected._compute_local_gradients()
        expected._mix_parameters(step=step)
        expected._apply_optimizer_update(expected._learning_rate(step))

    actual.train()

    assert torch.allclose(actual._local_vectors(), expected._local_vectors())
    assert not torch.allclose(
        actual._packed_storage_value("weight")[0],
        actual._packed_storage_value("weight")[1],
    )


def test_complete_mix_supports_partial_adaptive_gamma() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(
            topology=Topology.COMPLETE,
            mix=AdaptiveMixConfig(p=-1.0, min_gamma=0.25, max_gamma=1.0),
        ),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    with torch.no_grad():
        trainer._local_vectors()[0].fill_(0.0)
        trainer._local_vectors()[1].fill_(2.0)
        assert trainer.model is not None
        trainer.model.sync_parameters_from_storage_()

    trainer._mix_parameters(step=0, gamma=0.25)

    assert torch.allclose(
        trainer._local_vectors()[0],
        torch.full_like(trainer._local_vectors()[0], 0.25),
    )
    assert torch.allclose(
        trainer._local_vectors()[1],
        torch.full_like(trainer._local_vectors()[1], 1.75),
    )


def test_pairwise_adaptive_mix_does_not_mutate_before_lerp() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=4,
        trainer=DecentralizedTrainerConfig(
            topology=Topology.RING,
            mix=AdaptiveMixConfig(p=-1.0, min_gamma=0.5, max_gamma=1.0),
        ),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    original = torch.arange(4, dtype=trainer._local_vectors().dtype).unsqueeze(1)
    original = original.expand_as(trainer._local_vectors()).clone()
    with torch.no_grad():
        trainer._local_vectors().copy_(original)
        assert trainer.model is not None
        trainer.model.sync_parameters_from_storage_()

    trainer._mix_parameters(step=0, gamma=0.5)

    expected = original.clone()
    expected[0].fill_(0.25)
    expected[1].fill_(0.75)
    expected[2].fill_(2.25)
    expected[3].fill_(2.75)
    assert torch.allclose(trainer._local_vectors(), expected)


def test_ring_pairwise_peer_schedule_is_cached() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=4,
        trainer=DecentralizedTrainerConfig(topology=Topology.RING),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    assert len(trainer.pairwise_exchange_plans) == 2
    assert trainer._active_peer_by_rank(0) == {0: 1, 1: 0, 2: 3, 3: 2}
    assert trainer._active_peer_by_rank(1) == {0: 3, 1: 2, 2: 1, 3: 0}
    assert torch.equal(
        trainer.pairwise_exchange_plans[0].peer_local_indices,
        torch.tensor([1, 0, 3, 2]),
    )


def test_exp_pairwise_peer_schedule_is_cached() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=8,
        trainer=DecentralizedTrainerConfig(topology=Topology.EXP),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    assert len(trainer.pairwise_exchange_plans) == 3
    assert trainer._active_peer_by_rank(0)[0] == 1
    assert trainer._active_peer_by_rank(1)[0] == 2
    assert trainer._active_peer_by_rank(2)[0] == 4


def test_remote_pairwise_exchange_metadata_is_cached() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=4,
        trainer=DecentralizedTrainerConfig(topology=Topology.RING),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext(rank=0, world_size=2))

    local_phase, remote_phase = trainer.pairwise_exchange_plans
    assert local_phase.remote_processes == ()
    assert remote_phase.remote_processes == (1,)
    assert remote_phase.recv_by_process == {1: (2, 3)}
    assert torch.equal(
        remote_phase.send_local_indices_by_process[1],
        torch.tensor([0, 1]),
    )


def test_adaptive_gamma_schedule_matches_reference() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(
            topology=Topology.COMPLETE,
            mix=AdaptiveMixConfig(p=2.0, min_gamma=0.1, max_gamma=0.9, start_epoch=1),
        ),
        epochs=3,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        optimizer=OptimizerConfig(lr=0.2),
        scheduler=WarmupCosineSchedulerConfig(
            warmup_epochs=0,
            warmup_start_factor=1.0,
            eta_min_factor=0.0,
        ),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    start_step = trainer.batches_per_epoch
    start_lr = trainer._learning_rate(start_step)
    later_step = start_step + 1
    later_lr = trainer._learning_rate(later_step)

    assert trainer._mixing_gamma(0, trainer._learning_rate(0)) == pytest.approx(0.9)
    assert trainer._mixing_gamma(start_step, start_lr) == pytest.approx(0.9)
    expected = ((later_lr / start_lr) ** 2.0) * 0.8 + 0.1
    assert trainer._mixing_gamma(later_step, later_lr) == pytest.approx(expected)


def test_adaptive_negative_p_uses_min_gamma() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(
            topology=Topology.COMPLETE,
            mix=AdaptiveMixConfig(p=-1.0, min_gamma=0.2, max_gamma=0.9, start_epoch=1),
        ),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    assert trainer._mixing_gamma(0, trainer._learning_rate(0)) == pytest.approx(0.2)


def test_adaptive_gamma_one_matches_standard_decentralized() -> None:
    normal_cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(
            topology=Topology.COMPLETE,
            mix=NormalMixConfig(),
        ),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    adaptive_cfg = normal_cfg.model_copy(
        update={
            "trainer": DecentralizedTrainerConfig(
                topology=Topology.COMPLETE,
                mix=AdaptiveMixConfig(p=-1.0, min_gamma=1.0, max_gamma=1.0),
            )
        }
    )
    normal = DecentralizedTrainer(normal_cfg, ProcessContext())
    adaptive = DecentralizedTrainer(adaptive_cfg, ProcessContext())

    normal_metrics = normal.train()
    adaptive_metrics = adaptive.train()

    assert torch.allclose(normal._local_vectors(), adaptive._local_vectors())
    assert normal_metrics.loss == pytest.approx(adaptive_metrics.loss)


def test_complete_topology_uses_complete_mix_path(monkeypatch) -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    def fail_pairwise_mix(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("complete topology should not use pairwise mixing")

    monkeypatch.setattr(trainer, "_start_pairwise_topology_mix", fail_pairwise_mix)
    trainer.train()


def test_complete_mix_writes_directly_to_storage(monkeypatch) -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(
            topology=Topology.COMPLETE,
            mix=AdaptiveMixConfig(p=-1.0, min_gamma=0.25, max_gamma=1.0),
        ),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    with torch.no_grad():
        trainer._local_vectors()[0].fill_(0.0)
        trainer._local_vectors()[1].fill_(2.0)
        assert trainer.model is not None
        trainer.model.sync_parameters_from_storage_()

    def fail_load(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("complete mix should write directly to storage")

    monkeypatch.setattr(trainer, "_load_local_vectors_", fail_load)

    trainer._mix_parameters(step=0, gamma=0.25)

    assert torch.allclose(
        trainer._local_vectors()[0],
        torch.full_like(trainer._local_vectors()[0], 0.25),
    )
    assert torch.allclose(
        trainer._local_vectors()[1],
        torch.full_like(trainer._local_vectors()[1], 1.75),
    )


def test_sgd_update_uses_configured_momentum() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        optimizer=OptimizerConfig(lr=0.1, momentum=0.9, weight_decay=0.1),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    assert trainer.model is not None

    parameters = dict(trainer.model.named_parameters())
    weight = parameters["weight"]
    bias = parameters["bias"]
    with torch.no_grad():
        weight.fill_(2.0)
        bias.fill_(0.5)
    first_weight = weight.detach().clone()
    first_bias = bias.detach().clone()
    weight.grad = torch.full_like(weight, 0.25)
    bias.grad = torch.full_like(bias, -0.5)

    trainer._refresh_optimizer_gradients()
    trainer._apply_optimizer_update(lr=0.1)

    first_weight_update = torch.full_like(first_weight, 0.25).add(first_weight, alpha=0.1)
    first_bias_update = torch.full_like(first_bias, -0.5)
    expected_weight = first_weight.add(first_weight_update, alpha=-0.1)
    expected_bias = first_bias.add(first_bias_update, alpha=-0.1)
    assert torch.allclose(weight, expected_weight)
    assert torch.allclose(bias, expected_bias)

    second_weight = weight.detach().clone()
    second_bias = bias.detach().clone()
    weight.grad = torch.full_like(weight, 0.25)
    bias.grad = torch.full_like(bias, -0.5)
    trainer._refresh_optimizer_gradients()
    trainer._apply_optimizer_update(lr=0.1)

    second_weight_update = torch.full_like(second_weight, 0.25).add(second_weight, alpha=0.1)
    second_weight_buffer = first_weight_update.mul(0.9).add(second_weight_update)
    second_bias_buffer = first_bias_update.mul(0.9).add(-0.5)
    assert torch.allclose(weight, second_weight.add(second_weight_buffer, alpha=-0.1))
    assert torch.allclose(bias, second_bias.add(second_bias_buffer, alpha=-0.1))


def test_wide_resnet_cifar_trainer_smoke(monkeypatch) -> None:
    class FakeCifar:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            generator = torch.Generator(device="cpu")
            generator.manual_seed(123)
            self.images = torch.rand(16, 3, 32, 32, generator=generator)
            self.labels = torch.arange(16) % 10
            self.device = torch.device("cpu")

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
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del augment
            indices = deterministic_worker_indices(
                dataset_size=len(self),
                worker_rank=worker_rank,
                virtual_workers=virtual_workers,
                batch_size=batch_size,
                epoch=epoch,
                step=step,
                seed=seed,
                device=torch.device("cpu"),
                drop_last=True,
            )
            return self.images.index_select(0, indices), self.labels.index_select(0, indices)

    monkeypatch.setattr(trainer_module, "InMemoryCifar", FakeCifar)
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.WRN_16_1),
        optimizer=OptimizerConfig(lr=0.0),
        data=DataConfig(dataset=DatasetName.CIFAR10, batch_size=2, download=False),
    )
    metrics = DecentralizedTrainer(cfg, ProcessContext()).train()

    assert metrics.epochs == 1
    assert metrics.steps == 4
    assert metrics.owned_workers == (0, 1)
    assert torch.isfinite(torch.tensor(metrics.loss))


def test_warmup_epochs_are_converted_to_update_steps() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=2,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        optimizer=OptimizerConfig(lr=0.2),
        scheduler=WarmupCosineSchedulerConfig(
            warmup_epochs=1,
            warmup_start_factor=0.5,
            eta_min_factor=0.0,
        ),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    assert trainer.batches_per_epoch == 4
    assert trainer.total_steps == 8
    assert trainer._warmup_steps() == 4
    assert trainer._learning_rate(0) == 0.1
    assert trainer._learning_rate(4) == 0.2


def test_training_history_records_scheduled_evaluations() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=2,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    metrics = DecentralizedTrainer(cfg, ProcessContext()).train()

    assert len(metrics.history) == 1
    assert [item.epoch for item in metrics.history] == [2]
    assert all(torch.isfinite(torch.tensor(item.train_loss)) for item in metrics.history)
    assert all(torch.isfinite(torch.tensor(item.test_loss)) for item in metrics.history)
    assert all(0.0 <= item.test_accuracy <= 1.0 for item in metrics.history)
    assert metrics.history[-1].lr == metrics.lr


def test_bf16_amp_packed_cpu_smoke() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=4,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=True, amp_dtype="bf16"),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    metrics = trainer.train()

    assert trainer._amp_enabled()
    assert torch.isfinite(torch.tensor(metrics.loss))


def test_torch_compile_forward_path_is_used(monkeypatch) -> None:
    compile_calls = []

    def fake_compile(function, **kwargs):  # noqa: ANN001
        compile_calls.append(kwargs)

        def compiled(inputs: torch.Tensor) -> torch.Tensor:
            return function(inputs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False, compile=True, compile_mode="reduce-overhead"),
    )
    metrics = DecentralizedTrainer(cfg, ProcessContext()).train()

    assert compile_calls == [{"mode": "reduce-overhead", "fullgraph": False}]
    assert torch.isfinite(torch.tensor(metrics.loss))


def test_cuda_bf16_amp_wrn_smoke_uses_batched_autograd(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    class FakeCifar:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            device = kwargs.get("device", torch.device("cuda"))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(123)
            self.images = torch.rand(16, 3, 32, 32, generator=generator).to(device)
            self.labels = (torch.arange(16) % 10).long().to(device)
            self.device = device

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
        ) -> tuple[torch.Tensor, torch.Tensor]:
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

    monkeypatch.setattr(trainer_module, "InMemoryCifar", FakeCifar)
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cuda",
        model=ModelConfig(name=ModelName.WRN_16_1),
        optimizer=OptimizerConfig(lr=0.0),
        data=DataConfig(
            dataset=DatasetName.CIFAR10,
            batch_size=2,
            eval_batch_size=8,
            download=False,
        ),
        runtime=RuntimeConfig(amp=True, amp_dtype="bf16"),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    metrics = trainer.train()

    assert not trainer._use_cuda_amp_batched_autograd()
    assert trainer._amp_enabled()
    assert torch.isfinite(torch.tensor(metrics.loss))


def test_evaluation_uses_global_averaged_model() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.RING),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(
            dataset=DatasetName.SYNTHETIC,
            batch_size=2,
            eval_batch_size=8,
            num_classes=2,
        ),
        runtime=RuntimeConfig(amp=False),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    with torch.no_grad():
        trainer._packed_storage_value("weight")[0].fill_(1.0)
        trainer._packed_storage_value("weight")[1].fill_(-1.0)
        trainer._packed_storage_value("bias")[0].fill_(0.5)
        trainer._packed_storage_value("bias")[1].fill_(-0.5)
        assert trainer.model is not None
        trainer.model.sync_parameters_from_storage_()

    metrics = trainer._evaluate_epoch(epoch=1, train_loss=0.0, lr=0.0)

    assert torch.allclose(torch.tensor(metrics.test_loss), torch.log(torch.tensor(2.0)))


def test_evaluation_runs_every_five_epochs_and_final_epoch() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=7,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    evaluated_epochs = [
        epoch for epoch in range(1, cfg.epochs + 1) if trainer._should_evaluate_epoch(epoch)
    ]

    assert evaluated_epochs == [5, 7]


def test_evaluation_calibrates_batchnorm_before_testing(monkeypatch) -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(
            dataset=DatasetName.SYNTHETIC,
            batch_size=2,
            eval_batch_size=8,
            num_classes=2,
        ),
        runtime=RuntimeConfig(amp=False),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    calibrated_epochs = []

    def record_calibration(epoch: int) -> None:
        calibrated_epochs.append(epoch)

    monkeypatch.setattr(trainer, "_calibrate_average_model_batchnorm_", record_calibration)

    trainer._evaluate_epoch(epoch=1, train_loss=0.0, lr=0.0)

    assert calibrated_epochs == [1]


def test_eval_batch_size_is_capped_to_worker_shard_size() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(
            dataset=DatasetName.SYNTHETIC,
            batch_size=2,
            eval_batch_size=1024,
            num_classes=2,
        ),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    assert trainer.test_batch_size == 8
    assert trainer.test_batches_per_epoch == 1


def test_cli_cpu_smoke_single_process(tmp_path) -> None:
    config = tmp_path / "runtime.toml"
    config.write_text(
        """
[runtime]
compile = false
""",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "distributed_simulator.cli",
            str(config),
            "--dataset",
            "synthetic",
            "--model",
            "linear",
            "--workers",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--classes",
            "2",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "decentralized workers=2 processes=1" in result.stdout
    assert "epochs=1" in result.stdout


def test_cli_cpu_smoke_adaptive_mix(tmp_path) -> None:
    config = tmp_path / "adaptive.toml"
    config.write_text(
        """
[runtime]
compile = false

[trainer]
name = "decentralized"
topology = "complete"

[trainer.mix]
name = "adaptive"
start_epoch = 0
min_gamma = 0.5
max_gamma = 1.0
""",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "distributed_simulator.cli",
            str(config),
            "--dataset",
            "synthetic",
            "--model",
            "linear",
            "--workers",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--classes",
            "2",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "decentralized workers=2 processes=1" in result.stdout
    assert "mix=adaptive" in result.stdout
    assert "accum_gamma=" in result.stdout


def test_cli_sync_cpu_smoke_single_process(tmp_path) -> None:
    config = tmp_path / "sync.toml"
    config.write_text(
        """
[runtime]
compile = false

[trainer]
name = "sync"
""",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "distributed_simulator.cli",
            str(config),
            "--dataset",
            "synthetic",
            "--model",
            "linear",
            "--workers",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--classes",
            "2",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "sync workers=2 processes=1" in result.stdout
    assert "epochs=1" in result.stdout


def test_cli_cpu_smoke_torchrun_two_processes(tmp_path) -> None:
    config = tmp_path / "runtime.toml"
    config.write_text(
        """
[runtime]
compile = false
""",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "distributed_simulator.cli",
            str(config),
            "--dataset",
            "synthetic",
            "--model",
            "linear",
            "--workers",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--classes",
            "2",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "decentralized workers=4 processes=2" in result.stdout
    assert "epochs=1" in result.stdout


def test_cli_sync_cpu_smoke_torchrun_two_processes(tmp_path) -> None:
    config = tmp_path / "sync.toml"
    config.write_text(
        """
[runtime]
compile = false

[trainer]
name = "sync"
""",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "distributed_simulator.cli",
            str(config),
            "--dataset",
            "synthetic",
            "--model",
            "linear",
            "--workers",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--classes",
            "2",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "sync workers=4 processes=2" in result.stdout
    assert "epochs=1" in result.stdout
