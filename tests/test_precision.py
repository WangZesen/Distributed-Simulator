import pytest
import torch

from distributed_simulator.config import RuntimeConfig
from distributed_simulator.precision import configure_tf32


@pytest.mark.parametrize(
    ("runtime", "device", "expected"),
    [
        (RuntimeConfig(amp=False), torch.device("cuda"), True),
        (RuntimeConfig(amp=True, amp_dtype="bf16"), torch.device("cuda"), False),
        (RuntimeConfig(amp=False, tf32=False), torch.device("cuda"), False),
        (RuntimeConfig(amp=False), torch.device("cpu"), False),
    ],
)
def test_configure_tf32(
    monkeypatch: pytest.MonkeyPatch,
    runtime: RuntimeConfig,
    device: torch.device,
    expected: bool,
) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", not expected)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", not expected)

    assert configure_tf32(runtime, device) is expected
    assert torch.backends.cuda.matmul.allow_tf32 is expected
    assert torch.backends.cudnn.allow_tf32 is expected
