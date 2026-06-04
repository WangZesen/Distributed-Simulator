from __future__ import annotations

from torch import nn

NORM_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def get_param_groups(module: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay = []
    no_decay = []
    norm_parameter_names = _norm_parameter_names(module)
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_bias_name(name) or name in norm_parameter_names:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def parameter_decay_mask(module: nn.Module, weight_decay: float) -> dict[str, float]:
    norm_parameter_names = _norm_parameter_names(module)
    mask = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        mask[name] = 0.0 if _is_bias_name(name) or name in norm_parameter_names else weight_decay
    return mask


def _norm_parameter_names(module: nn.Module) -> set[str]:
    names = set()
    for module_name, child in module.named_modules():
        if isinstance(child, NORM_MODULES):
            for parameter_name, _ in child.named_parameters(recurse=False):
                names.add(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    return names


def _is_bias_name(name: str) -> bool:
    return name == "bias" or name.endswith(".bias")
