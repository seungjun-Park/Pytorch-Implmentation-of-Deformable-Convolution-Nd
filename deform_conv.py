import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

import custom_op
from utils import to_1tuple, to_2tuple, to_3tuple, multiply_integers


class DeformConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int]],
                 stride: Union[int, List[int], Tuple[int]] = 1,
                 padding: Union[int, List[int], Tuple[int]] = 0,
                 dilation: Union[int, List[int], Tuple[int]] = 1,
                 groups: int = 1,
                 deformable_groups_per_groups: int = 1,
                 bias: bool = True,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_1tuple(kernel_size)
        self.stride = to_1tuple(stride)
        self.padding = to_1tuple(padding)
        self.dilation = to_1tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_channels,
            ),
            nn.Conv1d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.attn_mask = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_channels,
            ),
            nn.Conv1d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)
        attn_mask = self.attn_mask(x)

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
            self.deformable_groups_per_groups,
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
                 deformable_groups_per_groups: int = 1,
                 bias: bool = True,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_channels,
            ),
            nn.Conv2d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes * 2,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.attn_mask = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_channels,
            ),
            nn.Conv2d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)
        attn_mask = self.attn_mask(x)

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
            self.deformable_groups_per_groups,
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
                 deformable_groups_per_groups: int = 1,
                 bias: bool = True,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_3tuple(kernel_size)
        self.stride = to_3tuple(stride)
        self.padding = to_3tuple(padding)
        self.dilation = to_3tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_channels,
            ),
            nn.Conv3d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes * 3,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.attn_mask = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                groups=in_channels,
            ),
            nn.Conv3d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x)
        attn_mask = self.attn_mask(x)

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
            self.deformable_groups_per_groups,
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
        raise NotImplementedError(f'{dim} is not supported.')
