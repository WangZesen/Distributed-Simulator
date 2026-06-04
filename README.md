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
