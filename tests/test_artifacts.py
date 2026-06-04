import csv
import tomllib
from datetime import datetime

import torch
from packed_resnet import WideResNet

import distributed_simulator.trainer as trainer_module
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
from distributed_simulator.trainer import DecentralizedTrainer
from distributed_simulator.trainers.base import EpochMetrics, TrainMetrics


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
    class FakeCifar:
        def __init__(self, *args, device: torch.device, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self.images = torch.randn(16, 3, 32, 32, device=device)
            self.labels = torch.arange(16, device=device) % 10

        def __len__(self) -> int:
            return self.images.size(0)

    monkeypatch.setattr(trainer_module, "InMemoryCifar", FakeCifar)
    cfg = DecentralizedConfig(
        virtual_workers=2,
        trainer=DecentralizedTrainerConfig(topology=Topology.COMPLETE),
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.WRN_16_1),
        data=DataConfig(dataset=DatasetName.CIFAR10, batch_size=2, download=False),
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
