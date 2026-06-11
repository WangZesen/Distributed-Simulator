from __future__ import annotations

import torch

from distributed_simulator.config import RuntimeConfig


def configure_tf32(runtime: RuntimeConfig, device: torch.device) -> bool:
    """Configure TF32 as the CUDA fallback when BF16 AMP is not requested."""
    enabled = device.type == "cuda" and runtime.tf32 and not runtime.amp
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    return enabled
