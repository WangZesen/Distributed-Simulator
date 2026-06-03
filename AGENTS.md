# Repository Guidelines

## Project Goal

This repository implements simulated distributed training for experiments where the target worker count is larger than the available physical GPU count. The implementation should let one or more GPUs simulate `N` virtual workers while preserving the algorithmic behavior of real distributed runs.

The number of virtual workers and the number of physical GPUs may be assumed to always be powers of two: `1`, `2`, `4`, and so on.

Primary algorithms to support:

- Synchronous SGD / data-parallel training.
- Sharpness-Aware Minimization (SAM).
- Decentralized training with topology-based parameter mixing.
- Decentralized variants using normal, adaptive, and random-sample mixing.

The implementation can ignore FWHT-based sampling and cyclic random-sample mixing.

The code under `reference/` is a reference implementation for real multi-GPU training. Treat it as behavioral guidance, not as the desired architecture for this simulator.

## Development Environment

- Use `uv` for virtual environment and dependency management.
- Prefer `uv run ...` for project commands and tests.
- Do not introduce dependency-management workflows based on `pip`, `conda`, Poetry, or Pipenv unless explicitly requested.

## Launch And Runtime

- Training jobs should be launchable with `torchrun`-style commands, including arguments such as `--nproc-per-node=x`.
- Keep launch-time process count separate from simulated virtual worker count. A process may simulate one or more virtual workers.
- Support CPU execution for smoke tests, both with a single CPU process and with multiple CPU processes/cores.
- Do not require CUDA, NCCL, or GPU-only APIs for basic correctness tests. Use CPU-compatible backends and code paths where practical.

## Implementation Principles

- Keep the simulator readable and professional: explicit data flow, clear object boundaries, and minimal hidden global state.
- Separate simulation concerns from algorithm concerns. For example, virtual worker state, communication/mixing schedules, topology definitions, optimizer steps, and metric collection should be independently understandable.
- Preserve semantics from the reference code where relevant, but avoid copying multi-process DDP structure directly when a single-process simulation abstraction is clearer.
- Make virtual worker count, physical device placement, topology, optimizer, scheduler, dataset, seed, AMP behavior, and logging configurable.
- Keep deterministic behavior practical: seed virtual workers, shuffling, augmentation, and communication sampling explicitly.
- Prefer typed configuration models and enums for structured options, following the style in `reference/conf.py`.
- Keep tensor operations batched/vectorized where possible, but do not sacrifice clarity for micro-optimizations before correctness is established.

## Reference Behavior To Preserve

- Model family: WideResNet variants from `reference/model.py` unless the user requests new models.
- Dataset baseline: CIFAR-10 and CIFAR-100 style loaders and transforms from `reference/data.py`.
- Optimizers: SGD, Adam, and AdamW with parameter groups that exclude bias and normalization-style 1D parameters from weight decay.
- Scheduler: cosine learning rate with optional warmup.
- Sync training: average gradients/updates across workers as in synchronous data parallel training.
- SAM: first ascent step, second loss/gradient evaluation, then restore base parameters before optimizer step.
- Decentralized training: simulate topology-specific communication groups and delayed `mix -> step -> start_comm` behavior where applicable.
- Decentralized metrics: preserve distance-to-consensus style evaluation when applicable.

## Code Quality Expectations

- Add tests for simulator behavior that can run on CPU where feasible; use tiny models/tensors rather than full training runs for algorithmic tests.
- Maintain smoke tests that run in CPU-only environments, including at least one single-process case and one multi-process launch case.
- Use GPU integration tests sparingly and keep them opt-in if they are expensive or environment-dependent.
- Avoid hard-coded cluster, SLURM, NCCL, or rank environment assumptions in simulator core code. Put environment-specific launch details at the edge.
- Keep logging and experiment output paths configurable. Avoid writing large artifacts by default in tests.
- Do not modify files under `reference/` unless the user explicitly asks to change the reference implementation.

## Useful Commands

Use these once project files exist:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
```

Adjust commands to match the actual `pyproject.toml` once it is added.
