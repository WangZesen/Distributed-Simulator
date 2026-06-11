import csv
import tomllib
from datetime import datetime

import torch
import torch.nn as nn
from packed_resnet import PackedDataLoader, WideResNet

import distributed_simulator.trainers.base as trainer_base
from distributed_simulator.artifacts import (
    create_run_dir,
    run_id_from_environment,
    save_last_checkpoints,
    save_resolved_config,
    save_stats_csv,
)
from distributed_simulator.config import (
    DataConfig,
    DecentralizedConfig,
    DecentralizedTrainerConfig,
    LoggingConfig,
    ModelConfig,
    SimulationConfig,
    Topology,
    config_from_files_and_overrides,
)
from distributed_simulator.data import DatasetName
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.model import ModelName
from distributed_simulator.trainers import DecentralizedTrainer, EpochMetrics, TrainMetrics


def _fake_packed_cifar_loader(*args, **kwargs) -> PackedDataLoader:  # noqa: ANN002
    del args
    device = kwargs["device"]
    return PackedDataLoader(
        torch.randn(16, 3, 32, 32, device=device),
        torch.arange(16, device=device) % 10,
        local_batch_size=kwargs["local_batch_size"],
        world_size=kwargs["world_size"],
        ranks=kwargs["ranks"],
        base_seed=kwargs["base_seed"],
        packed=True,
        channels_last=True,
        shuffle=kwargs["shuffle"],
        augment=False,
        normalize=False,
        sampler_drop_last=kwargs["sampler_drop_last"],
        drop_last=kwargs["drop_last"],
    )


def test_run_id_prefers_slurm_job_id(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOBID", "123456")

    assert run_id_from_environment(datetime(2026, 6, 4, 16, 42, 15)) == "123456"


def test_run_id_uses_concise_timestamp_without_slurm(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOBID", raising=False)

    assert run_id_from_environment(datetime(2026, 6, 4, 16, 42, 15)) == "20260604-164215"


def test_resolved_config_is_written_as_toml(tmp_path) -> None:
    cfg = SimulationConfig(logging=LoggingConfig(root=tmp_path / "logs"))
    run_dir = create_run_dir(cfg, run_id="run-1")

    save_resolved_config(cfg, run_dir / "config.toml")

    with (run_dir / "config.toml").open("rb") as file:
        data = tomllib.load(file)
    assert data["logging"]["root"] == str(tmp_path / "logs")
    assert data["logging"]["save_last_checkpoint"] is False
    assert data["runtime"]["compile"] is True


def test_stats_csv_writes_training_history(tmp_path) -> None:
    metrics = TrainMetrics(
        loss=1.0,
        distance_to_consensus=0.0,
        test_loss=2.0,
        test_accuracy=0.5,
        lr=0.1,
        gamma=0.0,
        accumulated_gamma=0.0,
        epochs=1,
        steps=4,
        rank=0,
        world_size=1,
        owned_workers=(0,),
        history=(
            EpochMetrics(
                epoch=1,
                train_loss=1.0,
                test_loss=2.0,
                test_accuracy=0.5,
                distance_to_consensus=0.0,
                lr=0.1,
                gamma=0.0,
                accumulated_gamma=0.0,
            ),
        ),
    )

    save_stats_csv(metrics, tmp_path / "stats.csv")

    with (tmp_path / "stats.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {
            "epoch": "1",
            "train_loss": "1.0",
            "test_loss": "2.0",
            "test_accuracy": "0.5",
            "distance_to_consensus": "0.0",
            "lr": "0.1",
            "gamma": "0.0",
            "accumulated_gamma": "0.0",
        }
    ]


def test_logging_config_parses_checkpoint_option(tmp_path) -> None:
    config = tmp_path / "logging.toml"
    config.write_text(
        f"""
[logging]
root = "{tmp_path / "runs"}"
save_last_checkpoint = true
""",
    )

    cfg = config_from_files_and_overrides([config], {})

    assert cfg.logging.root == tmp_path / "runs"
    assert cfg.logging.save_last_checkpoint is True


def test_decentralized_checkpoints_are_external_wide_resnet_state_dicts(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(trainer_base, "create_dataloader", _fake_packed_cifar_loader)
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.WRN_16_1),
        data=DataConfig(dataset=DatasetName.CIFAR10, batch_size=2),
        logging=LoggingConfig(root=tmp_path / "logs", save_last_checkpoint=True),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())

    save_last_checkpoints(trainer, tmp_path / "checkpoints")

    checkpoint_paths = sorted(tmp_path.glob("checkpoints/*.pth"))
    assert [path.name for path in checkpoint_paths] == [
        "global_last.pth",
        "local_worker_0.pth",
        "local_worker_1.pth",
    ]
    model = WideResNet(depth=16, widen_factor=1, num_classes=10)
    for path in checkpoint_paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)


def test_decentralized_checkpoints_retain_calibrated_global_batchnorm_buffers(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(trainer_base, "create_dataloader", _fake_packed_cifar_loader)
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.WRN_16_1),
        data=DataConfig(dataset=DatasetName.CIFAR10, batch_size=2, eval_batch_size=8),
        logging=LoggingConfig(root=tmp_path / "logs", save_last_checkpoint=True),
    )
    trainer = DecentralizedTrainer(cfg, ProcessContext())
    assert trainer.model is not None

    def set_calibrated_buffers(epoch: int) -> None:
        del epoch
        assert trainer.model is not None
        for module in trainer.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                assert module.running_mean is not None
                assert module.running_var is not None
                module.running_mean.fill_(7.0)
                module.running_var.fill_(3.0)

    monkeypatch.setattr(trainer, "_calibrate_average_model_batchnorm_", set_calibrated_buffers)

    trainer._evaluate_epoch(epoch=0, train_loss=0.0, lr=0.0)
    save_last_checkpoints(trainer, tmp_path / "checkpoints")

    for path in sorted((tmp_path / "checkpoints").glob("*.pth")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        for name, value in state.items():
            if name.endswith("running_mean"):
                assert torch.equal(value, torch.full_like(value, 7.0))
            elif name.endswith("running_var"):
                assert torch.equal(value, torch.full_like(value, 3.0))
