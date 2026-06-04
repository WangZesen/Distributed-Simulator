import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packed_resnet import WideResNet

import distributed_simulator.eval_hessian_eigenvalues as eval_module
from distributed_simulator.artifacts import save_resolved_config
from distributed_simulator.config import DataConfig, LoggingConfig, ModelConfig, SimulationConfig
from distributed_simulator.data import DatasetName
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.eval_hessian_eigenvalues import (
    _calibrate_batch_norm,
    _evaluation_batch,
    _load_model,
)
from distributed_simulator.eval_hessian_eigenvalues import (
    main as eval_hessian_main,
)
from distributed_simulator.hessian_eigenvalues import (
    HvpBatchSize,
    full_dataset_hvp,
    lanczos_eigenvalues,
)
from distributed_simulator.model import ModelName, get_model


def test_full_dataset_hvp_matches_explicit_autograd_average() -> None:
    torch.manual_seed(1)
    model = get_model(ModelName.LINEAR, num_classes=2)
    inputs = torch.randn(5, 3, 32, 32)
    targets = torch.tensor([0, 1, 0, 1, 1])
    parameters = list(model.parameters())
    vector = [torch.randn_like(parameter) for parameter in parameters]

    actual = full_dataset_hvp(
        model,
        (inputs, targets),
        HvpBatchSize(2),
        vector,
    )

    loss = F.cross_entropy(model(inputs), targets, reduction="sum") / targets.numel()
    grads = torch.autograd.grad(loss, parameters, create_graph=True)
    expected = torch.autograd.grad(grads, parameters, grad_outputs=vector)

    for actual_item, expected_item in zip(actual, expected, strict=True):
        assert torch.allclose(actual_item, expected_item, atol=1e-5, rtol=1e-5)


def test_lanczos_eigenvalues_returns_finite_ordered_extrema() -> None:
    torch.manual_seed(2)
    model = get_model(ModelName.LINEAR, num_classes=2)
    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.arange(8) % 2

    estimate = lanczos_eigenvalues(
        model,
        (inputs, targets),
        HvpBatchSize(4),
        num_iters=3,
        seed=123,
    )

    assert len(estimate.eigenvalues) == 3
    assert estimate.lambda_min <= estimate.lambda_max
    assert all(torch.isfinite(torch.tensor(value)) for value in estimate.eigenvalues)


def test_eval_hessian_eigenvalues_cli_writes_csv(tmp_path) -> None:
    run_dir = _write_linear_run(tmp_path)

    eval_hessian_main(
        [
            str(run_dir),
            "--device",
            "cpu",
            "--eval-batch-size",
            "4",
            "--num-eigenvalues",
            "2",
            "--bn-calibration",
            "none",
            "--eval-train-mode",
            "--augment-eval-data",
            "--log-level",
            "ERROR",
        ]
    )

    rows = _read_hessian_rows(run_dir / "hessian_eigenvalues.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["checkpoint"].endswith("checkpoints/global_last.pth")
    assert row["bn_calibration"] == "none"
    assert row["eval_train_mode"] == "True"
    assert row["augment_eval_data"] == "True"
    assert float(row["lambda_min"]) <= float(row["lambda_max"])
    eigenvalues = json.loads(row["eigenvalues"])
    assert len(eigenvalues) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in eigenvalues)


def test_eval_hessian_eigenvalues_rejects_train_mode_with_bn_calibration(tmp_path) -> None:
    run_dir = _write_linear_run(tmp_path)

    with pytest.raises(ValueError, match="--eval-train-mode requires --bn-calibration=none"):
        eval_hessian_main(
            [
                str(run_dir),
                "--device",
                "cpu",
                "--num-eigenvalues",
                "2",
                "--bn-calibration",
                "clean",
                "--eval-train-mode",
            ]
        )


def test_load_model_accepts_saved_external_wide_resnet_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "global_last.pth"
    cfg = SimulationConfig(
        virtual_workers=2,
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.WRN_16_1),
        data=DataConfig(dataset=DatasetName.CIFAR10, batch_size=2, download=False),
        logging=LoggingConfig(root=tmp_path / "logs"),
    )
    saved = WideResNet(depth=16, widen_factor=1, num_classes=10)
    torch.save(saved.state_dict(), checkpoint)

    loaded = _load_model(cfg, checkpoint, torch.device("cpu"))

    loaded.load_state_dict(saved.state_dict())


def test_batch_norm_calibration_updates_buffers_with_synthetic_data(tmp_path) -> None:
    cfg = SimulationConfig(
        virtual_workers=2,
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        logging=LoggingConfig(root=tmp_path / "logs"),
    )
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1),
        nn.BatchNorm2d(4),
        nn.Flatten(),
        nn.Linear(4 * 32 * 32, 2),
    )
    original_mean = model[1].running_mean.detach().clone()

    _calibrate_batch_norm(
        model,
        cfg,
        torch.device("cpu"),
        ProcessContext(),
        batch_size=2,
        augment=False,
    )

    assert not torch.allclose(model[1].running_mean, original_mean)
    assert model.training is False


def test_evaluation_batch_can_use_augmented_cifar_data(monkeypatch, tmp_path) -> None:
    class FakeCifar:
        def __init__(self, *args, device: torch.device, **kwargs) -> None:  # noqa: ANN002, ANN003
            del args, kwargs
            self.images = torch.arange(4 * 3 * 32 * 32, dtype=torch.float32, device=device).view(
                4,
                3,
                32,
                32,
            )
            self.labels = torch.arange(4, device=device)
            self.mean = torch.zeros(1, 3, 1, 1, device=device)
            self.std = torch.ones(1, 3, 1, 1, device=device)

    monkeypatch.setattr(eval_module, "InMemoryCifar", FakeCifar)
    cfg = SimulationConfig(
        virtual_workers=1,
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.WRN_16_1),
        data=DataConfig(
            dataset=DatasetName.CIFAR10,
            batch_size=2,
            num_classes=10,
            download=False,
        ),
        logging=LoggingConfig(root=tmp_path / "logs"),
    )

    clean_images, labels = _evaluation_batch(
        cfg,
        torch.device("cpu"),
        ProcessContext(),
        data_fraction=1.0,
        augment=False,
    )
    augmented_images, augmented_labels = _evaluation_batch(
        cfg,
        torch.device("cpu"),
        ProcessContext(),
        data_fraction=1.0,
        augment=True,
    )

    assert torch.equal(labels, augmented_labels)
    assert clean_images.shape == augmented_images.shape
    assert not torch.equal(clean_images, augmented_images)


def test_eval_hessian_eigenvalues_torchrun_two_processes(tmp_path) -> None:
    run_dir = _write_linear_run(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "distributed_simulator.eval_hessian_eigenvalues",
            str(run_dir),
            "--device",
            "cpu",
            "--eval-batch-size",
            "4",
            "--num-eigenvalues",
            "2",
            "--log-level",
            "ERROR",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "hessian_eigenvalues" in result.stdout
    rows = _read_hessian_rows(run_dir / "hessian_eigenvalues.csv")
    assert len(rows) == 1
    assert float(rows[0]["lambda_min"]) <= float(rows[0]["lambda_max"])


def _write_linear_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    cfg = SimulationConfig(
        virtual_workers=4,
        epochs=0,
        device="cpu",
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(
            dataset=DatasetName.SYNTHETIC,
            batch_size=2,
            num_classes=2,
        ),
        logging=LoggingConfig(root=tmp_path / "logs"),
    )
    save_resolved_config(cfg, run_dir / "config.toml")
    model = get_model(ModelName.LINEAR, num_classes=2)
    torch.save(model.state_dict(), checkpoint_dir / "global_last.pth")
    return run_dir


def _read_hessian_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))
