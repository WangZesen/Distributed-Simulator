import subprocess
import sys

import pytest
import torch

import distributed_simulator.trainer as trainer_module
from distributed_simulator.config import (
    DataConfig,
    DecentralizedConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    Topology,
    WarmupCosineSchedulerConfig,
)
from distributed_simulator.data import DatasetName, deterministic_worker_indices
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.model import ModelName
from distributed_simulator.trainer import DecentralizedTrainer


def test_standard_decentralized_cpu_smoke_single_process() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=4,
        topology=Topology.RING,
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
        topology=Topology.COMPLETE,
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=3, num_classes=4),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    images, labels = trainer._next_batch()

    assert images.shape == (3, 2, 3, 32, 32)
    assert labels.shape == (3, 2)


def test_decentralized_trainer_uses_packed_storage() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        topology=Topology.COMPLETE,
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


def test_complete_mix_happens_before_optimizer_update() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        topology=Topology.COMPLETE,
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


def test_complete_topology_uses_complete_mix_path(monkeypatch) -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        topology=Topology.COMPLETE,
        epochs=1,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    def fail_pairwise_mix(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("complete topology should not use pairwise mixing")

    monkeypatch.setattr(trainer, "_pairwise_topology_mix", fail_pairwise_mix)
    trainer.train()


def test_sgd_update_uses_configured_momentum() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        topology=Topology.COMPLETE,
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
        topology=Topology.COMPLETE,
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
        topology=Topology.COMPLETE,
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


def test_training_history_records_metrics_for_every_epoch() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        topology=Topology.COMPLETE,
        epochs=2,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
    )
    metrics = DecentralizedTrainer(cfg, ProcessContext()).train()

    assert len(metrics.history) == 2
    assert [item.epoch for item in metrics.history] == [1, 2]
    assert all(torch.isfinite(torch.tensor(item.train_loss)) for item in metrics.history)
    assert all(torch.isfinite(torch.tensor(item.test_loss)) for item in metrics.history)
    assert all(0.0 <= item.test_accuracy <= 1.0 for item in metrics.history)
    assert metrics.history[-1].lr == metrics.lr


def test_bf16_amp_packed_cpu_smoke() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=4,
        topology=Topology.COMPLETE,
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
        topology=Topology.COMPLETE,
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
        topology=Topology.COMPLETE,
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
        topology=Topology.RING,
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


def test_eval_batch_size_is_capped_to_worker_shard_size() -> None:
    cfg = DecentralizedConfig(
        virtual_workers=2,
        topology=Topology.COMPLETE,
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


def test_cli_cpu_smoke_single_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "distributed_simulator.cli",
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


def test_cli_cpu_smoke_torchrun_two_processes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "distributed_simulator.cli",
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
