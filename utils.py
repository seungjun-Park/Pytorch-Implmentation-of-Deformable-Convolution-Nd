import torch
import torch.nn as nn
from collections import abc

from itertools import repeat
from typing import Union, List, Tuple

def _ntuple(n):
    def parse(x) -> Tuple:
        if isinstance(x, abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def multiply_integers(x) -> int:
    mul = 1
    if not isinstance(x, abc.Iterable):
        return x
    for i in x:
        mul *= i

    return int(mul)


def group_norm(num_channels, num_groups=32, eps=1e-6, affine=True):
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


def get_act(name='relu', *args, **kwargs):
    name = name.lower()

    if name == 'relu':
        return nn.ReLU(*args, **kwargs)

    elif name == 'softplus':
        return nn.Softplus(*args, **kwargs)

    elif name == 'silu':
        return nn.SiLU(*args, **kwargs)

    elif name == 'sigmoid':
        return nn.Sigmoid(*args, **kwargs)

    elif name == 'tanh':
        return nn.Tanh(*args, **kwargs)

    elif name == 'hard_tanh':
        return nn.Hardtanh(*args, **kwargs)

    elif name == 'leaky_relu':
        return nn.LeakyReLU(*args, **kwargs)

    elif name == 'elu':
        return nn.ELU(*args, **kwargs)

    elif name == 'gelu':
        return nn.GELU(*args, **kwargs)

    elif name == 'mish':
        return nn.Mish(*args, **kwargs)

    else:
        NotImplementedError(f'Activation function "{name}" is not supported.')