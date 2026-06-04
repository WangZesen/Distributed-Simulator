import subprocess
import sys

import torch

import distributed_simulator.trainers.sam as sam_module
from distributed_simulator.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SAMTrainerConfig,
    SimulationConfig,
    config_from_files_and_overrides,
)
from distributed_simulator.data import DatasetName
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.model import ModelName
from distributed_simulator.trainer import SAMTrainer


def _sam_linear_config(**kwargs: object) -> SimulationConfig:
    values = {
        "virtual_workers": 4,
        "trainer": SAMTrainerConfig(),
        "epochs": 1,
        "device": "cpu",
        "model": ModelConfig(name=ModelName.LINEAR),
        "optimizer": OptimizerConfig(lr=0.1, momentum=0.0, weight_decay=0.0),
        "data": DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        "runtime": RuntimeConfig(amp=False, compile=False),
    }
    values.update(kwargs)
    return SimulationConfig(**values)


def test_sam_config_default_and_override(tmp_path) -> None:
    default_config = tmp_path / "sam-default.toml"
    default_config.write_text(
        """
[trainer]
name = "sam"
""",
    )
    override_config = tmp_path / "sam-override.toml"
    override_config.write_text(
        """
[trainer]
name = "sam"
rho = 0.12
""",
    )

    default_cfg = config_from_files_and_overrides([default_config], {})
    override_cfg = config_from_files_and_overrides([default_config, override_config], {})

    assert isinstance(default_cfg.trainer, SAMTrainerConfig)
    assert default_cfg.trainer.rho == 0.05
    assert isinstance(override_cfg.trainer, SAMTrainerConfig)
    assert override_cfg.trainer.rho == 0.12


def test_sam_cpu_smoke_single_process() -> None:
    metrics = SAMTrainer(_sam_linear_config(), ProcessContext()).train()

    assert metrics.epochs == 1
    assert metrics.steps == 4
    assert metrics.owned_workers == (0, 1, 2, 3)
    assert len(metrics.history) == 1
    assert torch.isfinite(torch.tensor(metrics.loss))
    assert torch.isfinite(torch.tensor(metrics.test_loss))
    assert 0.0 <= metrics.test_accuracy <= 1.0


def test_sam_restores_base_parameters_before_optimizer_step() -> None:
    cfg = _sam_linear_config(virtual_workers=2, epochs=0, trainer=SAMTrainerConfig(rho=0.05))
    trainer = SAMTrainer(cfg, ProcessContext())
    assert trainer.model is not None and trainer.param_storage is not None

    base_storage = trainer.param_storage.detach().clone()
    second_loss = trainer._compute_sam_gradients()
    second_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }

    assert torch.isfinite(second_loss)
    assert torch.allclose(trainer.param_storage, base_storage)
    for name, parameter in trainer.model.named_parameters():
        assert parameter.grad is not None
        assert torch.allclose(parameter.grad, second_gradients[name])

    trainer._average_gradients_()
    averaged_second_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    before_update = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }
    trainer._apply_optimizer_update(lr=0.1)

    for name, parameter in trainer.model.named_parameters():
        expected = before_update[name].add(averaged_second_gradients[name], alpha=-0.1)
        assert torch.allclose(parameter, expected)


def test_sam_training_keeps_replicas_in_consensus() -> None:
    trainer = SAMTrainer(_sam_linear_config(), ProcessContext())
    metrics = trainer.train()

    assert metrics.distance_to_consensus < 1e-6
    assert torch.allclose(
        trainer._local_vectors(),
        trainer._local_vectors()[0].expand_as(trainer._local_vectors()),
    )


def test_sam_gradient_averaging_uses_one_coalesced_all_reduce(monkeypatch) -> None:
    cfg = _sam_linear_config(virtual_workers=4, epochs=0)
    all_reduce_tensors = []

    def fake_broadcast(tensor: torch.Tensor, src: int) -> None:
        assert src == 0
        assert tensor.ndim == 1

    def fake_all_reduce(tensor: torch.Tensor, op: object) -> None:
        del op
        all_reduce_tensors.append(tensor.detach().clone())
        tensor.mul_(2)

    monkeypatch.setattr(sam_module.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(sam_module.dist, "all_reduce", fake_all_reduce)
    trainer = SAMTrainer(cfg, ProcessContext(rank=0, world_size=2))
    assert trainer.param_storage is not None

    trainer._compute_sam_gradients()
    trainer._average_gradients_()

    assert len(all_reduce_tensors) == 1
    assert all_reduce_tensors[0].shape == (trainer.param_storage.size(1),)


def test_cli_sam_cpu_smoke_single_process(tmp_path) -> None:
    config = tmp_path / "sam.toml"
    config.write_text(
        """
[runtime]
compile = false

[trainer]
name = "sam"
rho = 0.07
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
    assert "sam workers=2 processes=1" in result.stdout
    assert "rho=0.07" in result.stdout
