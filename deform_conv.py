import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_op

from collections import abc
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


def multiply_integers(x: abc.Iterable):
    if not isinstance(x, abc.Iterable):
        return x
    mul = 1
    for i in x:
        mul *= i
    return mul


def modulate(x: torch.Tensor, modulation_type: str = 'none', deformable_groups: int = None) -> torch.Tensor:
    modulation_type = modulation_type.lower()
    assert modulation_type in ['none', 'softmax', 'sigmoid', 'tanh']
    if modulation_type == 'none':
        return x
    elif modulation_type == 'softmax':
        assert deformable_groups is not None
        b, c, *spatial = x.shape
        return F.softmax(x.reshape(b, deformable_groups, -1, *spatial), dim=1).reshape(b, c, *spatial)
    elif modulation_type == 'sigmoid':
        return F.sigmoid(x)
    elif modulation_type == 'tanh':
        return F.tanh(x)
    else:
        NotImplementedError(f'{modulation_type} was not supported.')


class DeformConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 deformable_groups: int = 1,
                 bias: bool = True,
                 modulation_type: str = 'none',
                 kernel_size_off: Union[int, List[int], Tuple[int]] = None,
                 stride_off: Union[int, List[int], Tuple[int]] = None,
                 padding_off: Union[int, List[int], Tuple[int]] = None,
                 dilation_off: Union[int, List[int], Tuple[int]] = None,
                 groups_off: int = None,
                 bias_off: bool = None,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0 and in_channels % (groups * deformable_groups) == 0

        self.dim = 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_1tuple(kernel_size)
        self.stride = to_1tuple(stride)
        self.padding = to_1tuple(padding)
        self.dilation = to_1tuple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.modulation_type = modulation_type.lower()

        kernel_size_off = kernel_size if kernel_size_off is None else kernel_size_off
        stride_off = stride if stride_off is None else stride_off
        padding_off = padding if padding_off is None else padding_off
        dilation_off = dilation if dilation_off is None else dilation_off
        groups_off = groups if groups_off is None else groups_off
        bias_off = bias if bias_off is None else bias_off

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Conv1d(
            in_channels=in_channels,
            out_channels=groups * deformable_groups * kernel_sizes * self.dim,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.attn_mask = nn.Conv1d(
            in_channels=in_channels,
            out_channels=groups * deformable_groups * kernel_sizes * self.dim,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0.0, std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)
        attn_mask = self.attn_mask(x)
        attn_mask = modulate(attn_mask, self.modulation_type, self.deformable_groups)

        return torch.ops.custom_op.deform_conv1d(
            x,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.bias
        )


class DeformConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 deformable_groups: int = 1,
                 bias: bool = True,
                 modulation_type: str = 'none',
                 kernel_size_off: Union[int, List[int], Tuple[int]] = None,
                 stride_off: Union[int, List[int], Tuple[int]] = None,
                 padding_off: Union[int, List[int], Tuple[int]] = None,
                 dilation_off: Union[int, List[int], Tuple[int]] = None,
                 groups_off: int = None,
                 bias_off: bool = None,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0 and in_channels % (groups * deformable_groups) == 0

        self.dim = 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.modulation_type = modulation_type.lower()
        self.deformable_groups = deformable_groups

        kernel_size_off = kernel_size if kernel_size_off is None else kernel_size_off
        stride_off = stride if stride_off is None else stride_off
        padding_off = padding if padding_off is None else padding_off
        dilation_off = dilation if dilation_off is None else dilation_off
        groups_off = groups if groups_off is None else groups_off
        bias_off = bias if bias_off is None else bias_off

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Conv2d(
            in_channels=in_channels,
            out_channels=groups * deformable_groups * kernel_sizes * self.dim,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.attn_mask = nn.Conv2d(
            in_channels=in_channels,
            out_channels=groups * deformable_groups * kernel_sizes,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0.0, std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)
        attn_mask = self.attn_mask(x)
        attn_mask = modulate(attn_mask, self.modulation_type, self.deformable_groups)

        return torch.ops.custom_op.deform_conv2d(
            x,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.bias
        )


class DeformConv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 deformable_groups: int = 1,
                 bias: bool = True,
                 modulation_type: str = 'none',
                 kernel_size_off: Union[int, List[int], Tuple[int]] = None,
                 stride_off: Union[int, List[int], Tuple[int]] = None,
                 padding_off: Union[int, List[int], Tuple[int]] = None,
                 dilation_off: Union[int, List[int], Tuple[int]] = None,
                 groups_off: int = None,
                 bias_off: bool = None,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0 and in_channels % (groups * deformable_groups) == 0

        self.dim = 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_3tuple(kernel_size)
        self.stride = to_3tuple(stride)
        self.padding = to_3tuple(padding)
        self.dilation = to_3tuple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.modulation_type = modulation_type.lower()

        kernel_size_off = kernel_size if kernel_size_off is None else kernel_size_off
        stride_off = stride if stride_off is None else stride_off
        padding_off = padding if padding_off is None else padding_off
        dilation_off = dilation if dilation_off is None else dilation_off
        groups_off = groups if groups_off is None else groups_off
        bias_off = bias if bias_off is None else bias_off

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Conv3d(
            in_channels=in_channels,
            out_channels=groups * deformable_groups * kernel_sizes * self.dim,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.attn_mask = nn.Conv3d(
            in_channels=in_channels,
            out_channels=groups * deformable_groups * kernel_sizes,
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0.0, std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)
        attn_mask = self.attn_mask(x)
        attn_mask = modulate(attn_mask, self.modulation_type, self.deformable_groups)

        return torch.ops.custom_op.deform_conv3d(
            x,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.bias
        )


def deform_conv_nd(dim: int = 2, *args, **kwargs):
    if dim == 1:
        return DeformConv1d(*args, **kwargs)

    elif dim == 2:
        return DeformConv2d(*args, **kwargs)

    elif dim == 3:
        return DeformConv3d(*args, **kwargs)

    else:
        NotImplementedError("The dims should have a value between 1 and 3.")