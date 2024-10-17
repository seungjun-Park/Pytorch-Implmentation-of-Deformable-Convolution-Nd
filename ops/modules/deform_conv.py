import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple, Optional
from collections import abc

from modules.ops.functions.defom_conv_func import _deform_conv_nd
from .utils import to_ntuple, to_1tuple, to_2tuple, to_3tuple


def _check_dimension(target, dim: int) -> bool:
    if not isinstance(target, int):
        return len(target) == dim

    return True


def _multiply_integers(target: abc.Iterable) -> int:
    mul = 1

    for element in target:
        mul *= int(element)

    return int(mul)


def _apply_modulation_type(
        x: torch.Tensor,
        modulation_type: str = 'none',
        groups: int = None):

    if modulation_type == 'none':
        return x
    elif modulation_type == 'sigmoid':
        return x.sigmoid()
    elif modulation_type == 'softmax':
        assert groups is not None
        b, dim, c, *l = x.shape
        x = F.softmax(x.reshape(b, dim, groups, -1, *l), dim=2).reshape(b, dim, c, *l)
        return x
    else:
        NotImplementedError(f'apply {modulation_type} is not support.')


def deform_conv_1d(inps: torch.Tensor,
                   weight: torch.Tensor,
                   offset_field: torch.Tensor,
                   attn_mask: torch.Tensor,
                   kernel_size: Union[int, List[int], Tuple[int]],
                   stride: Union[int, List[int], Tuple[int]] = 1,
                   padding: Union[int, List[int], Tuple[int]] = 0,
                   dilation: Union[int, List[int], Tuple[int]] = 1,
                   groups: int = 1,
                   bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    if isinstance(kernel_size, int):
        kernel_size = to_1tuple(kernel_size)
    else:
        assert _check_dimension(kernel_size, dim=1)

    if isinstance(stride, int):
        stride = to_1tuple(stride)
    else:
        assert _check_dimension(stride, dim=1)

    if isinstance(padding, int):
        padding = to_1tuple(padding)
    else:
        assert _check_dimension(padding, dim=1)

    if isinstance(dilation, int):
        dilation = to_1tuple(dilation)
    else:
        assert _check_dimension(dilation, dim=1)

    return _deform_conv_nd(
        inps,
        weight,
        offset_field,
        attn_mask,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    )


def deform_conv_2d(inps: torch.Tensor,
                   weight: torch.Tensor,
                   offset_field: torch.Tensor,
                   attn_mask: torch.Tensor,
                   kernel_size: Union[int, List[int], Tuple[int]],
                   stride: Union[int, List[int], Tuple[int]],
                   padding: Union[int, List[int], Tuple[int]],
                   dilation: Union[int, List[int], Tuple[int]],
                   groups: int,
                   bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    if isinstance(kernel_size, int):
        kernel_size = to_2tuple(kernel_size)
    else:
        assert _check_dimension(kernel_size, dim=2)

    if isinstance(stride, int):
        stride = to_2tuple(stride)
    else:
        assert _check_dimension(stride, dim=2)

    if isinstance(padding, int):
        padding = to_2tuple(padding)
    else:
        assert _check_dimension(padding, dim=2)

    if isinstance(dilation, int):
        dilation = to_2tuple(dilation)
    else:
        assert _check_dimension(dilation, dim=2)

    return _deform_conv_nd(
        inps,
        weight,
        offset_field,
        attn_mask,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    )


def deform_conv_3d(inps: torch.Tensor,
                   weight: torch.Tensor,
                   offset_field: torch.Tensor,
                   attn_mask: torch.Tensor,
                   kernel_size: Union[int, List[int], Tuple[int]],
                   stride: Union[int, List[int], Tuple[int]],
                   padding: Union[int, List[int], Tuple[int]],
                   dilation: Union[int, List[int], Tuple[int]],
                   groups: int,
                   bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    if isinstance(kernel_size, int):
        kernel_size = to_3tuple(kernel_size)
    else:
        assert _check_dimension(kernel_size, dim=3)

    if isinstance(stride, int):
        stride = to_3tuple(stride)
    else:
        assert _check_dimension(stride, dim=3)

    if isinstance(padding, int):
        padding = to_3tuple(padding)
    else:
        assert _check_dimension(padding, dim=3)

    if isinstance(dilation, int):
        dilation = to_3tuple(dilation)
    else:
        assert _check_dimension(dilation, dim=3)

    return _deform_conv_nd(
        inps,
        weight,
        offset_field,
        attn_mask,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    )


class DeformConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: Union[int, List[int], Tuple[int]] = 1,
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

        assert in_channels % groups == 0 and out_channels % groups == 0
        assert modulation_type.lower() in ['none', 'sigmoid', 'softmax']

        self.modulation_type = modulation_type.lower()
        self.dim = 1

        self.kernel_size = to_1tuple(kernel_size)
        self.stride = to_1tuple(stride)
        self.padding = to_1tuple(padding)
        self.dilation = to_1tuple(dilation)
        self.groups = groups

        kernel_size_off = kernel_size_off if kernel_size_off else kernel_size
        stride_off = stride_off if stride_off else stride
        padding_off = padding_off if padding_off else padding
        dilation_off = dilation_off if dilation_off else dilation
        groups_off = groups_off if groups_off else groups
        bias_off = bias_off if bias_off else bias

        self.conv_off = nn.Conv1d(
            in_channels,
            in_channels * _multiply_integers(self.kernel_size) * (self.dim + 1),
            kernel_size_off,
            stride_off,
            padding_off,
            dilation_off,
            groups_off,
            bias_off
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self, std=0.02):
        self.weight.data.normal_(std=std)
        if self.bias is not None:
            self.bias.data.normal_(std=std)

    def forward(self, inps: torch.Tensor) -> torch.Tensor:
        off = self.conv_off(inps)
        b, c, l = off.shape
        off = off.reshape(b, self.dim + 1, -1, l)
        offset_field = off[:, 0: self.dim, :, :]
        attn_mask = off[:, self.dim: self.dim + 1, :, :]
        attn_mask = _apply_modulation_type(attn_mask, self.modulation_type, self.groups)

        return deform_conv_1d(
            inps,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias,
        )


class DeformConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: Union[int, List[int], Tuple[int]] = 1,
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

        assert in_channels % groups == 0 and out_channels % groups == 0
        assert modulation_type.lower() in ['none', 'sigmoid', 'softmax']

        self.modulation_type = modulation_type.lower()
        self.dim = 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.dilation = to_2tuple(dilation)
        self.groups = groups

        kernel_size_off = kernel_size_off if kernel_size_off is not None else kernel_size
        stride_off = stride_off if stride_off is not None else stride
        padding_off = padding_off if padding_off is not None else padding
        dilation_off = dilation_off if dilation_off is not None else dilation
        groups_off = groups_off if groups_off is not None else groups
        bias_off = bias_off if bias_off is not None else bias

        self.conv_off = nn.Conv2d(
            in_channels,
            in_channels * _multiply_integers(self.kernel_size) * (self.dim + 1),
            kernel_size=kernel_size_off,
            stride=stride_off,
            padding=padding_off,
            dilation=dilation_off,
            groups=groups_off,
            bias=bias_off
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size),
                                   requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self, std=0.02):
        self.weight.data.normal_(std=std)
        if self.bias is not None:
            self.bias.data.normal_(std=std)

    def forward(self, inps: torch.Tensor) -> torch.Tensor:
        off = self.conv_off(inps)
        b, c, h, w = off.shape
        off = off.reshape(b, self.dim + 1, -1, h, w)
        offset_field = off[:, 0: self.dim, :, :, :]
        attn_mask = off[:, self.dim: self.dim + 1, :, :, :]
        attn_mask = _apply_modulation_type(attn_mask, self.modulation_type, self.groups)

        return deform_conv_2d(
            inps,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias,
        )


class DeformConv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: Union[int, List[int], Tuple[int]] = 1,
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

        assert in_channels % groups == 0 and out_channels % groups == 0
        assert modulation_type.lower() in ['none', 'sigmoid', 'softmax']

        self.modulation_type = modulation_type.lower()
        self.dim = 3

        self.kernel_size = to_3tuple(kernel_size)
        self.stride = to_3tuple(stride)
        self.padding = to_3tuple(padding)
        self.dilation = to_3tuple(dilation)
        self.groups = groups

        kernel_size_off = kernel_size_off if kernel_size_off else kernel_size
        stride_off = stride_off if stride_off else stride
        padding_off = padding_off if padding_off else padding
        dilation_off = dilation_off if dilation_off else dilation
        groups_off = groups_off if groups_off else groups
        bias_off = bias_off if bias_off else bias

        self.conv_off = nn.Conv3d(
            in_channels,
            in_channels * _multiply_integers(self.kernel_size) * (self.dim + 1),
            kernel_size_off,
            stride_off,
            padding_off,
            dilation_off,
            groups_off,
            bias_off
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size),
                                   requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self, std=0.02):
        self.weight.data.normal_(std=std)
        if self.bias is not None:
            self.bias.data.normal_(std=std)

    def forward(self, inps: torch.Tensor) -> torch.Tensor:
        off = self.conv_off(inps)
        b, c, d, h, w = off.shape
        off = off.reshape(b, self.dim + 1, -1, d, h, w)
        offset_field = off[:, 0: self.dim, :, :, :, :]
        attn_mask = off[:, self.dim: self.dim + 1, :, :, :, :]
        attn_mask = _apply_modulation_type(attn_mask, self.modulation_type, self.groups)

        return deform_conv_3d(
            inps,
            self.weight,
            offset_field,
            attn_mask,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias,
        )

