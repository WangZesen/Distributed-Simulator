from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from distributed_simulator.data import DatasetName
from distributed_simulator.model import ModelName


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Topology(StrEnum):
    RING = "ring"
    EXP = "exp"
    COMPLETE = "complete"


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
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"


class DecentralizedConfig(_ConfigModel):
    virtual_workers: int = Field(default=8, gt=0)
    topology: Topology = Topology.RING
    epochs: int = Field(default=200, ge=0)
    seed: int = Field(default=42, ge=0)
    device: str = "cpu"
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=WarmupCosineSchedulerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("virtual_workers")
    @classmethod
    def validate_virtual_workers(cls, value: int) -> int:
        _require_power_of_two(value, "virtual_workers")
        return value

    @model_validator(mode="after")
    def validate_model_data_pair(self) -> DecentralizedConfig:
        if self.data.dataset != DatasetName.SYNTHETIC and self.model.name == ModelName.LINEAR:
            raise ValueError("CIFAR training requires a WideResNet model")
        if self.data.dataset == DatasetName.SYNTHETIC and self.model.name != ModelName.LINEAR:
            raise ValueError("WideResNet training requires CIFAR10 or CIFAR100 data")
        return self


def _require_power_of_two(value: int, name: str) -> None:
    if value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")
