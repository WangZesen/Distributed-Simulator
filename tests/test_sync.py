import torch

import distributed_simulator.trainer as trainer_module
from distributed_simulator.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SimulationConfig,
    SyncTrainerConfig,
)
from distributed_simulator.data import DatasetName
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.model import ModelName
from distributed_simulator.trainer import SyncTrainer


def test_sync_cpu_smoke_single_process() -> None:
    cfg = SimulationConfig(
        virtual_workers=4,
        trainer=SyncTrainerConfig(),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    metrics = SyncTrainer(cfg, ProcessContext()).train()

    assert metrics.epochs == 1
    assert metrics.steps == 4
    assert metrics.owned_workers == (0, 1, 2, 3)
    assert len(metrics.history) == 1
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert torch.isfinite(torch.tensor(metrics.test_loss))
    assert 0.0 <= metrics.test_accuracy <= 1.0


def test_sync_averages_gradients_before_optimizer_step() -> None:
    cfg = SimulationConfig(
        virtual_workers=2,
        trainer=SyncTrainerConfig(),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        optimizer=OptimizerConfig(lr=0.1, momentum=0.0, weight_decay=0.0),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False),
    )
    trainer = SyncTrainer(cfg, ProcessContext())
    assert trainer.model is not None

    original = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }
    trainer._compute_local_gradients()
    raw_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }

    trainer._average_gradients_()
    for name, parameter in trainer.model.named_parameters():
        assert parameter.grad is not None
        expected = raw_gradients[name].reshape(2, -1).mean(dim=0)
        expected = expected.expand_as(parameter.grad.reshape(2, -1)).reshape_as(parameter.grad)
        assert torch.allclose(parameter.grad, expected)

    trainer._apply_optimizer_update(lr=0.1)
    for name, parameter in trainer.model.named_parameters():
        expected_gradient = raw_gradients[name].reshape(2, -1).mean(dim=0)
        expected_gradient = expected_gradient.expand_as(parameter.reshape(2, -1))
        expected_gradient = expected_gradient.reshape_as(parameter)
        assert torch.allclose(parameter, original[name].add(expected_gradient, alpha=-0.1))


def test_sync_training_keeps_replicas_in_consensus() -> None:
    cfg = SimulationConfig(
        virtual_workers=4,
        trainer=SyncTrainerConfig(),
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False),
    )
    trainer = SyncTrainer(cfg, ProcessContext())
    metrics = trainer.train()

    assert metrics.distance_to_consensus < 1e-6
    assert torch.allclose(
        trainer._local_vectors(),
        trainer._local_vectors()[0].expand_as(trainer._local_vectors()),
    )


def test_sync_gradient_averaging_uses_one_coalesced_all_reduce(monkeypatch) -> None:
    cfg = SimulationConfig(
        virtual_workers=4,
        trainer=SyncTrainerConfig(),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False),
    )
    all_reduce_tensors = []

    def fake_broadcast(tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        assert tensor.ndim == 1

    def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
        del op
        all_reduce_tensors.append(tensor.detach().clone())
        tensor.mul_(2)

    monkeypatch.setattr(trainer_module.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(trainer_module.dist, "all_reduce", fake_all_reduce)
    trainer = SyncTrainer(cfg, ProcessContext(rank=0, world_size=2))
    assert trainer.model is not None and trainer.param_storage is not None

    trainer._compute_local_gradients()
    raw_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }

    trainer._average_gradients_()

    assert len(all_reduce_tensors) == 1
    assert all_reduce_tensors[0].shape == (trainer.param_storage.size(1),)
    for name, parameter in trainer.model.named_parameters():
        assert parameter.grad is not None
        expected = raw_gradients[name].reshape(trainer.local_worker_count, -1).mean(dim=0)
        expected = expected.expand_as(parameter.grad.reshape(trainer.local_worker_count, -1))
        expected = expected.reshape_as(parameter.grad)
        assert torch.allclose(parameter.grad, expected)
