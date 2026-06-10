from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple, Self

import torch
import torch.nn as nn
from packed_resnet import MLP, PackedMLP, PackedWideResNet, WideResNet

_PARAMETER_STORAGE_ALIGNMENT = 64
_IMAGE_FEATURES = 3 * 32 * 32


class ModelName(StrEnum):
    LINEAR = "linear"
    WRN_16_1 = "WRN_16_1"
    WRN_16_8 = "WRN_16_8"
    WRN_16_10 = "WRN_16_10"
    WRN_28_10 = "WRN_28_10"
    WRN_28_12 = "WRN_28_12"
    WRN_40_1 = "WRN_40_1"
    WRN_40_2 = "WRN_40_2"
    WRN_40_4 = "WRN_40_4"


class PackedParameterLayout(NamedTuple):
    name: str
    module_name: str
    parameter_name: str
    start: int
    numel: int
    padded_numel: int
    shape: tuple[int, ...]


def _align_numel(numel: int) -> int:
    remainder = numel % _PARAMETER_STORAGE_ALIGNMENT
    return numel if remainder == 0 else numel + _PARAMETER_STORAGE_ALIGNMENT - remainder


def packed_parameter_view(tensor: torch.Tensor, num_models: int) -> torch.Tensor:
    """Return a zero-copy physical-layout ``[K, D]`` view of a packed tensor."""
    local_numel = tensor.numel() // num_models
    if tensor.is_contiguous():
        return tensor.view(num_models, local_numel)
    if tensor.ndim == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
        return tensor.as_strided((num_models, local_numel), (local_numel, 1))
    raise ValueError(
        f"packed tensor shape {tuple(tensor.shape)} must be contiguous or channels-last contiguous"
    )


def parameter_storage_layout(
    module: nn.Module, num_models: int
) -> tuple[PackedParameterLayout, ...]:
    entries = []
    offset = 0
    for name, parameter in module.named_parameters():
        local_numel = parameter.numel() // num_models
        local_shape = (
            tuple(parameter.shape[1:]) if parameter.shape[0] == num_models else (local_numel,)
        )
        module_name, _, parameter_name = name.rpartition(".")
        entries.append(
            PackedParameterLayout(
                name=name,
                module_name=module_name,
                parameter_name=parameter_name,
                start=offset,
                numel=local_numel,
                padded_numel=_align_numel(local_numel),
                shape=local_shape,
            )
        )
        offset += _align_numel(local_numel)
    return tuple(entries)


class ImageLinearClassifier(MLP):
    def __init__(self, num_classes: int):
        super().__init__(
            in_features=_IMAGE_FEATURES,
            hidden_features=(),
            out_features=num_classes,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs.flatten(start_dim=1))


class PackedImageLinearClassifier(PackedMLP):
    """Upstream packed linear model with simulator parameter-mixing storage."""

    def __init__(self, num_models: int, num_classes: int):
        super().__init__(
            num_models=num_models,
            in_features=_IMAGE_FEATURES,
            hidden_features=(),
            out_features=num_classes,
        )
        self.num_classes = num_classes
        self._parameter_storage: torch.Tensor | None = None
        self._layout = parameter_storage_layout(self, num_models)
        self._parameter_views: list[torch.Tensor] = []
        self._storage_views: list[torch.Tensor] = []
        self._bind_storage_views()

    @property
    def weight(self) -> nn.Parameter:
        return self.layers[0].weight

    @property
    def bias(self) -> nn.Parameter:
        bias = self.layers[0].bias
        assert bias is not None
        return bias

    @property
    def parameter_storage(self) -> torch.Tensor:
        if self._parameter_storage is None:
            parameter = next(self.parameters())
            self._parameter_storage = torch.zeros(
                self.num_models,
                self.parameter_storage_numel(),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            self._bind_storage_views()
            self.sync_storage_from_parameters_()
        return self._parameter_storage

    def parameter_storage_numel(self) -> int:
        return sum(item.padded_numel for item in self._layout)

    def parameter_storage_layout(self) -> tuple[PackedParameterLayout, ...]:
        return self._layout

    def _bind_storage_views(self) -> None:
        parameters = dict(self.named_parameters())
        self._parameter_views = [
            packed_parameter_view(parameters[item.name], self.num_models) for item in self._layout
        ]
        self._storage_views = (
            [
                self._parameter_storage[:, item.start : item.start + item.numel]
                for item in self._layout
            ]
            if self._parameter_storage is not None
            else []
        )

    def _apply(self, fn, recurse: bool = True) -> Self:  # noqa: ANN001
        result = super()._apply(fn, recurse)
        self._parameter_storage = None
        self._bind_storage_views()
        return result

    def sync_storage_from_parameters_(self) -> Self:
        storage = self.parameter_storage
        del storage
        with torch.no_grad():
            torch._foreach_copy_(self._storage_views, self._parameter_views)
        return self

    def sync_parameters_from_storage_(self) -> Self:
        storage = self.parameter_storage
        del storage
        with torch.no_grad():
            torch._foreach_copy_(self._parameter_views, self._storage_views)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs.flatten(start_dim=2))


def get_model(model_name: ModelName, num_classes: int) -> nn.Module:
    if model_name == ModelName.LINEAR:
        return ImageLinearClassifier(num_classes=num_classes)
    depth, width_factor = _wide_resnet_dimensions(model_name)
    return WideResNet(depth=depth, widen_factor=width_factor, num_classes=num_classes)


def get_packed_model(model_name: ModelName, num_classes: int, num_models: int) -> nn.Module:
    if model_name == ModelName.LINEAR:
        return PackedImageLinearClassifier(num_models=num_models, num_classes=num_classes)
    depth, width_factor = _wide_resnet_dimensions(model_name)
    return PackedWideResNet(
        depth=depth,
        widen_factor=width_factor,
        num_models=num_models,
        num_classes=num_classes,
    )


def _wide_resnet_dimensions(model_name: ModelName) -> tuple[int, int]:
    return int(model_name.value.split("_")[1]), int(model_name.value.split("_")[-1])
