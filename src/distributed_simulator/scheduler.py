from __future__ import annotations

import math

from distributed_simulator.config import ConstantSchedulerConfig, WarmupCosineSchedulerConfig

SchedulerConfig = ConstantSchedulerConfig | WarmupCosineSchedulerConfig


def lr_factor(
    cfg: SchedulerConfig,
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
) -> float:
    if total_steps <= 0:
        return 1.0
    if cfg.name == "constant":
        return 1.0
    if cfg.name == "warmup_cosine":
        warmup_steps = min(warmup_steps, total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            progress = step / warmup_steps
            return cfg.warmup_start_factor + (1.0 - cfg.warmup_start_factor) * progress
        cosine_steps = max(total_steps - warmup_steps, 1)
        cosine_step = min(max(step - warmup_steps, 0), cosine_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_steps))
        return cfg.eta_min_factor + (1.0 - cfg.eta_min_factor) * cosine
    raise ValueError(f"unsupported scheduler: {cfg.name}")
