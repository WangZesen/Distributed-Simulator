# Distributed Simulator

Simulated distributed training for experiments where the number of virtual workers can exceed the number of physical devices.

The first implemented trainers are standard decentralized training and synchronous SGD,
with CPU smoke-test support and `torchrun`-style launches. Synthetic and CIFAR data both
use image-shaped batches; the linear smoke-test model flattens images internally.

```bash
uv sync --extra dev
uv run pytest
uv run torchrun --standalone --nproc-per-node=2 -m distributed_simulator.cli --dataset synthetic --model linear --workers 4 --epochs 1 --device cpu
```

Training configuration can be supplied as one or more TOML files. Files are merged in
order, with later files overriding earlier files; smoke-test flags such as `--device`,
`--workers`, `--epochs`, `--dataset`, `--model`, and `--batch-size` override files.

Example sync trainer config:

```toml
[trainer]
name = "sync"
```

```bash
uv run torchrun --standalone --nproc-per-node=2 -m distributed_simulator.cli sync.toml --dataset synthetic --model linear --workers 4 --epochs 1 --device cpu
```

Example CIFAR/WideResNet launch:

```bash
uv run torchrun --standalone --nproc-per-node=1 -m distributed_simulator.cli cifar_wrn.toml --workers 2 --epochs 10 --device cuda
```

CIFAR data is loaded into the target device memory before training. Sampling and augmentation are deterministic with respect to the configured seed, epoch, and virtual worker rank.

Profile the real CIFAR batch path without adding it to the default pytest suite:

```bash
uv run python scripts/profile_cifar_dataloader.py --datasets CIFAR10 CIFAR100 --device cpu
```

Profile trainer phases after warmup:

```bash
uv run dsim-profile-trainer config/sync.toml --dataset synthetic --model linear --workers 4 --device cpu
uv run torchrun --standalone --nproc-per-node=2 -m distributed_simulator.profile_trainers config/sync.toml --dataset synthetic --model linear --workers 4 --device cpu
```

Use `--warmup-steps`, `--profile-steps`, `--profile-evaluation`, and `--json-output`
to control the profile. CUDA phase timings synchronize the device, while the separately
reported end-to-end step retains configured communication overlap.

Compare only forward, loss, and backward against an isolated Packed-ResNet baseline:

```bash
uv run dsim-profile-forward-backward config/sync.toml --dataset CIFAR10 --model WRN_16_8 --workers 8 --device cuda --batch-size 16
```

This profiler uses the same random packed batch for both cases and enables
`torch.backends.cudnn.benchmark` by default to match
`external/Packed-ResNet/tests/benchmark_gpu_timing.py`.

All trainers enable `torch.backends.cudnn.benchmark` by default. Set
`runtime.cudnn_benchmark = false` when deterministic cuDNN algorithm selection is required.

CUDA training uses BF16 AMP when `runtime.amp = true`. When AMP is disabled, CUDA training
uses TF32 by default for matrix multiplications and cuDNN operations. Set
`runtime.tf32 = false` to retain full FP32 precision.

All trainers use fused SGD by default. Set `optimizer.fused = false` to use the standard
optimizer implementation.
