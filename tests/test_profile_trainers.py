import pytest

from distributed_simulator.config import (
    DataConfig,
    DecentralizedTrainerConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SAMTrainerConfig,
    SimulationConfig,
    SyncTrainerConfig,
)
from distributed_simulator.data import DatasetName
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.model import ModelName
from distributed_simulator.profile_trainers import profile_trainer
from distributed_simulator.trainers import DecentralizedTrainer, SAMTrainer, SyncTrainer


@pytest.mark.parametrize(
    ("trainer_config", "trainer_type"),
    [
        (SyncTrainerConfig(), SyncTrainer),
        (SAMTrainerConfig(), SAMTrainer),
        (DecentralizedTrainerConfig(), DecentralizedTrainer),
    ],
)
def test_all_trainers_use_fused_optimizer_by_default(trainer_config, trainer_type) -> None:
    cfg = SimulationConfig(
        virtual_workers=2,
        epochs=0,
        device="cpu",
        trainer=trainer_config,
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False, compile=False),
    )

    trainer = trainer_type(cfg, ProcessContext())

    assert trainer.optimizer is not None
    assert trainer.optimizer.defaults["fused"] is True


def test_fused_optimizer_can_be_disabled() -> None:
    cfg = SimulationConfig(
        virtual_workers=2,
        epochs=0,
        device="cpu",
        trainer=SyncTrainerConfig(),
        model=ModelConfig(name=ModelName.LINEAR),
        optimizer=OptimizerConfig(fused=False),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False, compile=False),
    )

    trainer = SyncTrainer(cfg, ProcessContext())

    assert trainer.optimizer is not None
    assert trainer.optimizer.defaults["fused"] is False


@pytest.mark.parametrize(
    ("trainer_config", "trainer_type", "expected_phases"),
    [
        (
            SyncTrainerConfig(),
            SyncTrainer,
            {"batch", "forward_backward", "gradient_average", "optimizer_storage_sync"},
        ),
        (
            SAMTrainerConfig(),
            SAMTrainer,
            {"batch", "sam_forward_backward", "gradient_average", "optimizer_storage_sync"},
        ),
        (
            DecentralizedTrainerConfig(overlap_mixing=False),
            DecentralizedTrainer,
            {"batch", "forward_backward", "mix", "optimizer_storage_sync"},
        ),
    ],
)
def test_profile_trainer_reports_trainer_specific_phases(
    trainer_config,
    trainer_type,
    expected_phases: set[str],
) -> None:
    cfg = SimulationConfig(
        virtual_workers=2,
        epochs=1,
        device="cpu",
        trainer=trainer_config,
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False, compile=False),
    )
    trainer = trainer_type(cfg, ProcessContext())

    profile = profile_trainer(
        trainer,
        warmup_steps=0,
        profile_steps=1,
        profile_evaluation=False,
    )

    assert profile.trainer == trainer_config.name
    assert profile.parameter_storage_mb > 0
    assert profile.cold_batch_ms >= 0
    assert set(profile.phases) == expected_phases | {"end_to_end_step"}
    assert all(stats.mean_ms >= 0 for stats in profile.phases.values())
