from distributed_simulator.config import (
    DataConfig,
    ModelConfig,
    RuntimeConfig,
    SimulationConfig,
    SyncTrainerConfig,
)
from distributed_simulator.data import DatasetName
from distributed_simulator.distributed import ProcessContext
from distributed_simulator.model import ModelName
from distributed_simulator.profile_forward_backward import profile_forward_backward


def test_profile_forward_backward_compares_baseline_and_trainer_paths() -> None:
    cfg = SimulationConfig(
        virtual_workers=2,
        epochs=1,
        device="cpu",
        trainer=SyncTrainerConfig(),
        model=ModelConfig(name=ModelName.LINEAR),
        data=DataConfig(dataset=DatasetName.SYNTHETIC, batch_size=2, num_classes=2),
        runtime=RuntimeConfig(amp=False, compile=False),
    )

    profile = profile_forward_backward(
        cfg,
        ProcessContext(),
        warmup_steps=0,
        profile_steps=1,
        cudnn_benchmark=False,
    )

    assert profile.local_workers == 2
    assert profile.input_shape == (2, 2, 3, 32, 32)
    assert not profile.channels_last
    assert set(profile.cases) == {
        "packed_resnet_baseline",
        "trainer_forward_model",
        "trainer_compute_local_gradients",
    }
    assert set(profile.cases["packed_resnet_baseline"].phases) == {
        "forward",
        "loss",
        "backward",
        "total",
    }
    assert set(profile.cases["trainer_compute_local_gradients"].phases) == {"total"}
