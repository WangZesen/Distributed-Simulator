from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import NamedTuple, Self, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from packed_resnet import PackedWideResNet


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


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.equal_in_out = in_planes == out_planes
        self.conv_shortcut = None
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
        out = self.relu1(self.bn1(x)) if self.equal_in_out else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if not self.equal_in_out and self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_planes: int,
        out_planes: int,
        stride: int,
    ):
        super().__init__()
        layers = []
        for layer_idx in range(num_layers):
            layers.append(
                BasicBlock(
                    in_planes if layer_idx == 0 else out_planes,
                    out_planes,
                    stride if layer_idx == 0 else 1,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        num_classes: int,
        width_factor: int = 1,
    ):
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WideResNet depth must satisfy (depth - 4) % 6 == 0, got {depth}")
        num_layers = (depth - 4) // 6
        channels = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(num_layers, channels[0], channels[1], 1)
        self.block2 = NetworkBlock(num_layers, channels[1], channels[2], 2)
        self.block3 = NetworkBlock(num_layers, channels[2], channels[3], 2)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.channels)
        return self.fc(out)


class LinearClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(start_dim=1))


class PackedParameterLayout(NamedTuple):
    name: str
    module_name: str
    parameter_name: str
    start: int
    numel: int
    padded_numel: int
    shape: tuple[int, ...]
    is_view: bool


class PackedLinearClassifier(nn.Module):
    """Independent linear classifiers packed across local virtual workers."""

    parameter_storage: torch.Tensor
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(self, num_models: int, num_classes: int):
        super().__init__()
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}")
        self.num_models = num_models
        self.num_classes = num_classes
        self.in_features = 3 * 32 * 32
        weight_numel = num_classes * self.in_features
        bias_numel = num_classes
        self._parameter_storage_layout = (
            PackedParameterLayout(
                name="weight",
                module_name="",
                parameter_name="weight",
                start=0,
                numel=weight_numel,
                padded_numel=weight_numel,
                shape=(num_classes, self.in_features),
                is_view=True,
            ),
            PackedParameterLayout(
                name="bias",
                module_name="",
                parameter_name="bias",
                start=weight_numel,
                numel=bias_numel,
                padded_numel=bias_numel,
                shape=(num_classes,),
                is_view=True,
            ),
        )
        self.register_buffer(
            "parameter_storage",
            torch.empty(num_models, self.parameter_storage_numel()),
        )
        self._bind_parameter_storage_views()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        bound = 1 / self.in_features**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def parameter_storage_numel(self) -> int:
        return sum(item.padded_numel for item in self._parameter_storage_layout)

    def parameter_storage_layout(self) -> tuple[PackedParameterLayout, ...]:
        return self._parameter_storage_layout

    def sync_storage_from_parameters_(self) -> PackedLinearClassifier:
        return self

    def sync_parameters_from_storage_(self, include_conv: bool = True) -> PackedLinearClassifier:
        del include_conv
        return self

    def _bind_parameter_storage_views(self) -> None:
        weight, bias = self._parameter_storage_layout
        parameter_storage = cast(torch.Tensor, self.parameter_storage)
        self.weight = nn.Parameter(
            parameter_storage[:, weight.start : weight.start + weight.numel].view(
                self.num_models,
                *weight.shape,
            )
        )
        self.bias = nn.Parameter(
            parameter_storage[:, bias.start : bias.start + bias.numel].view(
                self.num_models,
                *bias.shape,
            )
        )

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        result = super()._apply(fn, recurse)
        self._bind_parameter_storage_views()
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"PackedLinearClassifier expects [B, K, C, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.num_models:
            raise ValueError(f"expected K={self.num_models}, got {x.shape[1]}")
        features = x.flatten(start_dim=2)
        return torch.einsum("bkf,kof->bko", features, self.weight) + self.bias.unsqueeze(0)


def get_model(
    model_name: ModelName,
    num_classes: int,
) -> nn.Module:
    if model_name == ModelName.LINEAR:
        return LinearClassifier(num_classes=num_classes)
    depth = int(model_name.value.split("_")[1])
    width_factor = int(model_name.value.split("_")[-1])
    return WideResNet(
        depth=depth,
        num_classes=num_classes,
        width_factor=width_factor,
    )


def get_packed_model(
    model_name: ModelName,
    num_classes: int,
    num_models: int,
) -> nn.Module:
    if model_name == ModelName.LINEAR:
        return PackedLinearClassifier(num_models=num_models, num_classes=num_classes)
    depth = int(model_name.value.split("_")[1])
    width_factor = int(model_name.value.split("_")[-1])
    return PackedWideResNet(
        depth=depth,
        widen_factor=width_factor,
        num_models=num_models,
        num_classes=num_classes,
    )
