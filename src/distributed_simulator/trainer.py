from distributed_simulator.trainers import (
    DecentralizedTrainer,
    EpochMetrics,
    SyncTrainer,
    TrainMetrics,
)
from distributed_simulator.trainers import base as _base

# Compatibility exports for tests and callers that monkeypatch these symbols through
# distributed_simulator.trainer. New code should import algorithm trainers from
# distributed_simulator.trainers.
dist = _base.dist
InMemoryCifar = _base.InMemoryCifar
InMemorySyntheticImages = _base.InMemorySyntheticImages

__all__ = [
    "DecentralizedTrainer",
    "EpochMetrics",
    "InMemoryCifar",
    "InMemorySyntheticImages",
    "SyncTrainer",
    "TrainMetrics",
    "dist",
]
