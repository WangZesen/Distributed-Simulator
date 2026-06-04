from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from distributed_simulator.data import DatasetName
from distributed_simulator.model import ModelName


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Topology(StrEnum):
    RING = "ring"
    EXP = "exp"
    COMPLETE = "complete"


class TrainerName(StrEnum):
    DECENTRALIZED = "decentralized"
    SYNC = "sync"
    SAM = "sam"


class OptimizerConfig(_ConfigModel):
    lr: float = Field(default=0.1, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0)
    weight_decay: float = Field(default=5e-4, ge=0.0)


class ModelConfig(_ConfigModel):
    name: ModelName = ModelName.WRN_16_8


class ConstantSchedulerConfig(_ConfigModel):
    name: Literal["constant"] = "constant"


class WarmupCosineSchedulerConfig(_ConfigModel):
    name: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_epochs: int = Field(default=10, ge=0)
    warmup_start_factor: float = Field(default=0.1, ge=0.0, le=1.0)
    eta_min_factor: float = Field(default=0.0, ge=0.0, le=1.0)


SchedulerConfig = Annotated[
    ConstantSchedulerConfig | WarmupCosineSchedulerConfig,
    Field(discriminator="name"),
]


class DataConfig(_ConfigModel):
    dataset: DatasetName = DatasetName.CIFAR10
    root: Path = Path("data")
    download: bool = True
    augment: bool = True
    num_classes: int = Field(default=10, gt=1)
    batch_size: int = Field(default=16, gt=0)
    eval_batch_size: int = Field(default=10000, gt=0)
    seed: int = Field(default=1234, ge=0)


class RuntimeConfig(_ConfigModel):
    amp: bool = True
    amp_dtype: Literal["bf16"] = "bf16"
    compile: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"


class LoggingConfig(_ConfigModel):
    root: Path = Path("logs")
    save_last_checkpoint: bool = False


class NormalMixConfig(_ConfigModel):
    name: Literal["normal"] = "normal"


class AdaptiveMixConfig(_ConfigModel):
    name: Literal["adaptive"] = "adaptive"
    p: float = 3.0
    max_gamma: float = Field(default=1.0, ge=0.0)
    min_gamma: float = Field(default=0.0, ge=0.0)
    start_epoch: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def validate_gamma_range(self) -> AdaptiveMixConfig:
        if self.min_gamma > self.max_gamma:
            raise ValueError("min_gamma must be less than or equal to max_gamma")
        return self


MixConfig = Annotated[
    NormalMixConfig | AdaptiveMixConfig,
    Field(discriminator="name"),
]


class DecentralizedTrainerConfig(_ConfigModel):
    name: Literal["decentralized"] = "decentralized"
    topology: Topology = Topology.RING
    overlap_mixing: bool = True
    mix: MixConfig = Field(default_factory=NormalMixConfig)


class SyncTrainerConfig(_ConfigModel):
    name: Literal["sync"] = "sync"


class SAMTrainerConfig(_ConfigModel):
    name: Literal["sam"] = "sam"
    rho: float = Field(default=0.05, ge=0.0)


TrainerConfig = DecentralizedTrainerConfig | SyncTrainerConfig | SAMTrainerConfig


class SimulationConfig(_ConfigModel):
    virtual_workers: int = Field(default=8, gt=0)
    epochs: int = Field(default=200, ge=0)
    seed: int = Field(default=42, ge=0)
    device: str = "cpu"
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=WarmupCosineSchedulerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    trainer: TrainerConfig = Field(default_factory=DecentralizedTrainerConfig, discriminator="name")

    @field_validator("virtual_workers")
    @classmethod
    def validate_virtual_workers(cls, value: int) -> int:
        _require_power_of_two(value, "virtual_workers")
        return value

    @model_validator(mode="after")
    def validate_model_data_pair(self) -> SimulationConfig:
        if self.data.dataset != DatasetName.SYNTHETIC and self.model.name == ModelName.LINEAR:
            raise ValueError("CIFAR training requires a WideResNet model")
        if self.data.dataset == DatasetName.SYNTHETIC and self.model.name != ModelName.LINEAR:
            raise ValueError("WideResNet training requires CIFAR10 or CIFAR100 data")
        return self


class DecentralizedConfig(SimulationConfig):
    @model_validator(mode="before")
    @classmethod
    def migrate_flat_decentralized_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "topology" not in data and "overlap_mixing" not in data:
            return data

        migrated = dict(data)
        trainer = migrated.get("trainer")
        trainer_data = (
            trainer.model_dump() if isinstance(trainer, _ConfigModel) else dict(trainer or {})
        )
        trainer_data.setdefault("name", "decentralized")
        if "topology" in migrated:
            trainer_data["topology"] = migrated.pop("topology")
        if "overlap_mixing" in migrated:
            trainer_data["overlap_mixing"] = migrated.pop("overlap_mixing")
        migrated["trainer"] = trainer_data
        return migrated

    @model_validator(mode="after")
    def validate_decentralized_trainer(self) -> DecentralizedConfig:
        if not isinstance(self.trainer, DecentralizedTrainerConfig):
            raise ValueError("DecentralizedConfig requires a decentralized trainer config")
        return self

    @property
    def topology(self) -> Topology:
        if not isinstance(self.trainer, DecentralizedTrainerConfig):
            raise ValueError("DecentralizedConfig requires a decentralized trainer config")
        return self.trainer.topology

    @property
    def overlap_mixing(self) -> bool:
        if not isinstance(self.trainer, DecentralizedTrainerConfig):
            raise ValueError("DecentralizedConfig requires a decentralized trainer config")
        return self.trainer.overlap_mixing


def _require_power_of_two(value: int, name: str) -> None:
    if value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def merge_dicts_recursive(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge config dictionaries with later values taking precedence."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict) and not _changes_named_block(
            current,
            value,
        ):
            merged[key] = merge_dicts_recursive(current, value)
        else:
            merged[key] = value
    return merged


def load_config_files(paths: list[str | Path] | tuple[str | Path, ...]) -> SimulationConfig:
    return config_from_files_and_overrides(paths, {})


def config_from_files_and_overrides(
    paths: list[str | Path] | tuple[str | Path, ...],
    overrides: dict[str, Any],
) -> SimulationConfig:
    data: dict[str, Any] = {}
    for path in paths:
        with Path(path).open("rb") as file:
            data = merge_dicts_recursive(data, tomllib.load(file))
    data = merge_dicts_recursive(data, overrides)
    return SimulationConfig.model_validate(data)


def _changes_named_block(current: dict[str, Any], override: dict[str, Any]) -> bool:
    return "name" in current and "name" in override and current["name"] != override["name"]
