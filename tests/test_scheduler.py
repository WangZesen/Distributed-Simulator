from distributed_simulator.config import ConstantSchedulerConfig, WarmupCosineSchedulerConfig
from distributed_simulator.scheduler import lr_factor


def test_constant_scheduler_factor() -> None:
    cfg = ConstantSchedulerConfig()
    assert lr_factor(cfg, step=0, total_steps=10) == 1.0
    assert lr_factor(cfg, step=9, total_steps=10) == 1.0


def test_warmup_cosine_scheduler_factor() -> None:
    cfg = WarmupCosineSchedulerConfig(
        warmup_epochs=1,
        warmup_start_factor=0.2,
        eta_min_factor=0.1,
    )
    assert lr_factor(cfg, step=0, total_steps=6, warmup_steps=2) == 0.2
    assert lr_factor(cfg, step=2, total_steps=6, warmup_steps=2) == 1.0
    assert lr_factor(cfg, step=6, total_steps=6, warmup_steps=2) == 0.1
