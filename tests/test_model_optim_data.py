import torch
from torch import nn

from distributed_simulator import data as data_module
from distributed_simulator.config import (
    AdaptiveMixConfig,
    DecentralizedConfig,
    DecentralizedTrainerConfig,
    NormalMixConfig,
    SimulationConfig,
    SyncTrainerConfig,
    Topology,
    WarmupCosineSchedulerConfig,
    config_from_files_and_overrides,
    merge_dicts_recursive,
)
from distributed_simulator.data import (
    DatasetName,
    InMemoryCifar,
    InMemorySyntheticImages,
    deterministic_cifar_augment,
    deterministic_epoch_order,
    deterministic_worker_indices,
    worker_indices_from_order,
)
from distributed_simulator.model import (
    ModelName,
    get_model,
    get_packed_model,
    packed_parameter_view,
)
from distributed_simulator.optim import get_param_groups, parameter_decay_mask


def test_linear_classifier_flattens_image_input() -> None:
    model = get_model(ModelName.LINEAR, num_classes=7)
    outputs = model(torch.randn(5, 3, 32, 32))
    assert outputs.shape == (5, 7)


def test_wide_resnet_forward_shape() -> None:
    model = get_model(ModelName.WRN_16_1, num_classes=10)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 32, 32).contiguous(memory_format=torch.channels_last))
    assert outputs.shape == (2, 10)


def test_wide_resnet_has_no_dropout_modules() -> None:
    model = get_model(ModelName.WRN_16_1, num_classes=10)
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())


def test_packed_model_initializes_identically_with_isolated_lazy_storage() -> None:
    model = get_packed_model(ModelName.LINEAR, num_classes=3, num_models=4)
    assert model._parameter_storage is None

    for parameter in model.parameters():
        local_parameters = packed_parameter_view(parameter, 4)
        assert torch.equal(local_parameters, local_parameters[0].expand_as(local_parameters))

    storage = model.parameter_storage
    parameter_ptrs = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
    assert storage.untyped_storage().data_ptr() not in parameter_ptrs

    storage.zero_()
    assert any(torch.count_nonzero(parameter) for parameter in model.parameters())
    model.sync_parameters_from_storage_()
    assert all(not torch.count_nonzero(parameter) for parameter in model.parameters())


def test_weight_decay_excludes_bias_and_batch_norm_parameters() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, bias=False),
        nn.BatchNorm2d(4),
        nn.Flatten(),
        nn.Linear(3600, 10),
    )
    mask = parameter_decay_mask(model, weight_decay=0.1)
    assert mask["0.weight"] == 0.1
    assert mask["1.weight"] == 0.0
    assert mask["1.bias"] == 0.0
    assert mask["3.weight"] == 0.1
    assert mask["3.bias"] == 0.0

    groups = get_param_groups(model, weight_decay=0.1)
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0


def test_worker_indices_are_deterministic_and_disjoint() -> None:
    first = deterministic_worker_indices(
        dataset_size=32,
        worker_rank=0,
        virtual_workers=4,
        batch_size=4,
        epoch=2,
        step=1,
        seed=123,
        device=torch.device("cpu"),
        drop_last=True,
    )
    second = deterministic_worker_indices(
        dataset_size=32,
        worker_rank=0,
        virtual_workers=4,
        batch_size=4,
        epoch=2,
        step=1,
        seed=123,
        device=torch.device("cpu"),
        drop_last=True,
    )
    other_worker = deterministic_worker_indices(
        dataset_size=32,
        worker_rank=1,
        virtual_workers=4,
        batch_size=4,
        epoch=2,
        step=1,
        seed=123,
        device=torch.device("cpu"),
        drop_last=True,
    )
    assert torch.equal(first, second)
    assert set(first.tolist()).isdisjoint(other_worker.tolist())


def test_worker_indices_can_reuse_epoch_order() -> None:
    order = deterministic_epoch_order(
        dataset_size=32,
        epoch=2,
        seed=123,
        device=torch.device("cpu"),
    )
    cached = worker_indices_from_order(
        order=order,
        worker_rank=0,
        virtual_workers=4,
        batch_size=4,
        step=1,
        drop_last=True,
    )
    direct = deterministic_worker_indices(
        dataset_size=32,
        worker_rank=0,
        virtual_workers=4,
        batch_size=4,
        epoch=2,
        step=1,
        seed=123,
        device=torch.device("cpu"),
        drop_last=True,
    )
    assert torch.equal(cached, direct)


def test_deterministic_cifar_augment_depends_on_epoch_and_step() -> None:
    images = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32).view(2, 3, 32, 32)
    first = deterministic_cifar_augment(images, seed=7, epoch=0, worker_rank=0)
    second = deterministic_cifar_augment(images, seed=7, epoch=0, worker_rank=0)
    next_epoch = deterministic_cifar_augment(images, seed=7, epoch=1, worker_rank=0)
    next_step = deterministic_cifar_augment(images, seed=7, epoch=0, step=1, worker_rank=0)
    assert torch.equal(first, second)
    assert not torch.equal(first, next_epoch)
    assert not torch.equal(first, next_step)


def test_in_memory_cifar_loads_full_split_to_target_device(monkeypatch) -> None:
    raw_images = torch.arange(16 * 3 * 32 * 32, dtype=torch.uint8).view(16, 3, 32, 32)
    raw_labels = torch.arange(16, dtype=torch.long) % 10

    def fake_load(*args, **kwargs):  # noqa: ANN002, ANN003
        return raw_images, raw_labels

    monkeypatch.setattr(data_module, "_load_cifar_tensors", fake_load)
    loader = InMemoryCifar(
        DatasetName.CIFAR10,
        root="unused",
        train=True,
        device=torch.device("cpu"),
    )

    assert loader.images.device.type == "cpu"
    assert loader.labels.device.type == "cpu"
    assert loader.images.shape == (16, 3, 32, 32)
    first_images, first_labels = loader.batch_for_worker(
        worker_rank=0,
        virtual_workers=2,
        batch_size=4,
        epoch=3,
        step=1,
        seed=9,
        augment=False,
    )
    second_images, second_labels = loader.batch_for_worker(
        worker_rank=0,
        virtual_workers=2,
        batch_size=4,
        epoch=3,
        step=1,
        seed=9,
        augment=False,
    )
    assert torch.equal(first_images, second_images)
    assert torch.equal(first_labels, second_labels)

    multi_images, multi_labels = loader.batch_for_workers(
        worker_ranks=(0, 1),
        virtual_workers=2,
        batch_size=4,
        epoch=3,
        step=1,
        seed=9,
        augment=False,
    )
    worker_one_images, worker_one_labels = loader.batch_for_worker(
        worker_rank=1,
        virtual_workers=2,
        batch_size=4,
        epoch=3,
        step=1,
        seed=9,
        augment=False,
    )
    assert torch.equal(multi_images[:, 0], first_images)
    assert torch.equal(multi_labels[:, 0], first_labels)
    assert torch.equal(multi_images[:, 1], worker_one_images)
    assert torch.equal(multi_labels[:, 1], worker_one_labels)


def test_batched_cifar_augmentation_matches_per_worker_path(monkeypatch) -> None:
    raw_images = torch.arange(32 * 3 * 32 * 32, dtype=torch.uint8).view(32, 3, 32, 32)
    raw_labels = torch.arange(32, dtype=torch.long) % 10

    def fake_load(*args, **kwargs):  # noqa: ANN002, ANN003
        return raw_images, raw_labels

    monkeypatch.setattr(data_module, "_load_cifar_tensors", fake_load)
    loader = InMemoryCifar(
        DatasetName.CIFAR10,
        root="unused",
        train=True,
        device=torch.device("cpu"),
    )

    multi_images, multi_labels = loader.batch_for_workers(
        worker_ranks=(0, 1, 2, 3),
        virtual_workers=4,
        batch_size=4,
        epoch=2,
        step=1,
        seed=9,
        augment=True,
    )

    for worker_rank in range(4):
        worker_images, worker_labels = loader.batch_for_worker(
            worker_rank=worker_rank,
            virtual_workers=4,
            batch_size=4,
            epoch=2,
            step=1,
            seed=9,
            augment=True,
        )
        assert torch.equal(multi_images[:, worker_rank], worker_images)
        assert torch.equal(multi_labels[:, worker_rank], worker_labels)


def test_batched_synthetic_data_matches_per_worker_path() -> None:
    loader = InMemorySyntheticImages(
        samples=64,
        num_classes=3,
        seed=123,
        device=torch.device("cpu"),
    )

    multi_images, multi_labels = loader.batch_for_workers(
        worker_ranks=(0, 1, 2, 3),
        virtual_workers=4,
        batch_size=4,
        epoch=2,
        step=1,
        seed=9,
        augment=False,
    )

    for worker_rank in range(4):
        worker_images, worker_labels = loader.batch_for_worker(
            worker_rank=worker_rank,
            virtual_workers=4,
            batch_size=4,
            epoch=2,
            step=1,
            seed=9,
            augment=False,
        )
        assert torch.equal(multi_images[:, worker_rank], worker_images)
        assert torch.equal(multi_labels[:, worker_rank], worker_labels)


def test_default_training_configuration_matches_requested_setup() -> None:
    cfg = SimulationConfig()
    assert cfg.model.name == ModelName.WRN_16_8
    assert cfg.data.dataset == DatasetName.CIFAR10
    assert cfg.virtual_workers == 8
    assert cfg.epochs == 200
    assert cfg.optimizer.lr == 0.1
    assert isinstance(cfg.scheduler, WarmupCosineSchedulerConfig)
    assert cfg.scheduler.warmup_epochs == 10
    assert cfg.data.batch_size == 16
    assert cfg.data.eval_batch_size == 10000
    assert cfg.runtime.amp is True
    assert cfg.runtime.amp_dtype == "bf16"
    assert cfg.runtime.compile is True
    assert cfg.runtime.compile_mode == "default"
    assert cfg.logging.root.as_posix() == "logs"
    assert cfg.logging.save_last_checkpoint is False
    assert cfg.optimizer.momentum == 0.9
    assert cfg.optimizer.weight_decay == 5e-4
    assert isinstance(cfg.trainer, DecentralizedTrainerConfig)
    assert cfg.trainer.topology == Topology.RING
    assert cfg.trainer.overlap_mixing is True
    assert isinstance(cfg.trainer.mix, NormalMixConfig)


def test_decentralized_config_accepts_legacy_flat_topology() -> None:
    cfg = DecentralizedConfig.model_validate({"topology": Topology.COMPLETE})

    assert cfg.topology == Topology.COMPLETE


def test_decentralized_config_accepts_adaptive_mix() -> None:
    cfg = DecentralizedConfig.model_validate(
        {
            "trainer": {
                "name": "decentralized",
                "topology": "complete",
                "mix": {
                    "name": "adaptive",
                    "p": 2.0,
                    "max_gamma": 0.8,
                    "min_gamma": 0.2,
                    "start_epoch": 3,
                },
            }
        }
    )

    assert isinstance(cfg.trainer, DecentralizedTrainerConfig)
    assert isinstance(cfg.trainer.mix, AdaptiveMixConfig)
    assert cfg.trainer.mix.p == 2.0
    assert cfg.trainer.mix.max_gamma == 0.8
    assert cfg.trainer.mix.min_gamma == 0.2
    assert cfg.trainer.mix.start_epoch == 3


def test_merge_dicts_recursive_replaces_named_blocks_when_name_changes() -> None:
    merged = merge_dicts_recursive(
        {
            "trainer": {
                "name": "decentralized",
                "topology": "ring",
                "mix": {"name": "adaptive", "p": 2.0},
            }
        },
        {"trainer": {"name": "sync"}},
    )

    assert merged == {"trainer": {"name": "sync"}}


def test_config_files_merge_in_order_and_cli_overrides_apply_last(tmp_path) -> None:
    base = tmp_path / "base.toml"
    override = tmp_path / "override.toml"
    base.write_text(
        """
virtual_workers = 8
epochs = 5
device = "cuda"

[model]
name = "linear"

[data]
dataset = "synthetic"
batch_size = 4
num_classes = 3

[trainer]
name = "decentralized"
topology = "ring"
""",
    )
    override.write_text(
        """
epochs = 2

[trainer]
name = "sync"
""",
    )

    cfg = config_from_files_and_overrides(
        [base, override],
        {"device": "cpu", "data": {"batch_size": 2}},
    )

    assert cfg.virtual_workers == 8
    assert cfg.epochs == 2
    assert cfg.device == "cpu"
    assert cfg.data.batch_size == 2
    assert isinstance(cfg.trainer, SyncTrainerConfig)
