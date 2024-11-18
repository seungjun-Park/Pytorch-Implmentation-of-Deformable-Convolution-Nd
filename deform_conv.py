import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

import custom_op
from utils import to_1tuple, to_2tuple, to_3tuple, multiply_integers, group_norm, get_act


def get_modulation_scalar(
        x: torch.Tensor,
        modulation_type: str = 'none',
        groups: int = None) -> torch.Tensor:
    modulation_type = modulation_type.lower()

    if modulation_type == 'none':
        return x

    elif modulation_type == 'softmax':
        assert groups is not None
        b, c, *spatial = x.shape
        return F.softmax(x.reshape(b, groups, -1, *spatial), dim=2).reshape(b, c, *spatial)

    elif modulation_type == 'sigmoid':
        return F.sigmoid(x)

    elif modulation_type == 'tanh':
        return F.tanh(x)

    else:
        NotImplementedError(f'{modulation_type} is not implemented.')


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
                 offset_scale: float = 1.0,
                 modulation_type: str = 'none',
                 bias: bool = True,
                 dw_kernel_size: Union[int, List[int], Tuple[int]] = 7,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_1tuple(kernel_size)
        self.stride = to_1tuple(stride)
        self.padding = to_1tuple(padding)
        self.dilation = to_1tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups
        self.offset_scale = offset_scale
        self.modulation_type = modulation_type

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                padding=(dw_kernel_size - 1) // 2,
                groups=in_channels,
            ),
            group_norm(in_channels, num_groups=in_channels),
            nn.Conv3d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes * 3,
                kernel_size=1,
            )
        )

        self.attn_mask = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                padding=(dw_kernel_size - 1) // 2,
                groups=in_channels,
            ),
            group_norm(in_channels, num_groups=in_channels),
            nn.Conv3d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes,
                kernel_size=1,
            )
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

        nn.init.zeros_(self.offset_field[0].weight)
        nn.init.zeros_(self.offset_field[-1].weight)
        nn.init.zeros_(self.offset_field[0].bias)
        nn.init.zeros_(self.offset_field[-1].bias)

        nn.init.zeros_(self.attn_mask[0].weight)
        nn.init.zeros_(self.attn_mask[-1].weight)
        nn.init.zeros_(self.attn_mask[0].bias)
        nn.init.ones_(self.attn_mask[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x) * self.offset_scale
        attn_mask = self.attn_mask(x)
        attn_mask = get_modulation_scalar(attn_mask, self.modulation_type, self.groups * self.deformable_groups_per_groups)

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
                 offset_scale: float = 1.0,
                 modulation_type: str = 'none',
                 bias: bool = True,
                 dw_kernel_size: Union[int, List[int], Tuple[int]] = 7,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups
        self.offset_scale = offset_scale
        self.modulation_type = modulation_type

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                padding=(dw_kernel_size - 1) // 2,
                groups=in_channels,
            ),
            group_norm(in_channels, num_groups=in_channels),
            nn.Conv2d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes * 2,
                kernel_size=1,
            )
        )

        self.attn_mask = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                padding=(dw_kernel_size - 1) // 2,
                groups=in_channels,
            ),
            group_norm(in_channels, num_groups=in_channels),
            nn.Conv2d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes,
                kernel_size=1,
            )
        )

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

        nn.init.zeros_(self.offset_field[0].weight)
        nn.init.zeros_(self.offset_field[-1].weight)
        nn.init.zeros_(self.offset_field[0].bias)
        nn.init.zeros_(self.offset_field[-1].bias)

        nn.init.zeros_(self.attn_mask[0].weight)
        nn.init.zeros_(self.attn_mask[-1].weight)
        nn.init.zeros_(self.attn_mask[0].bias)
        nn.init.ones_(self.attn_mask[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x) * self.offset_scale
        attn_mask = self.attn_mask(x)
        attn_mask = get_modulation_scalar(attn_mask, self.modulation_type, self.groups * self.deformable_groups_per_groups)

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
                 offset_scale: float = 1.0,
                 modulation_type: str = 'none',
                 bias: bool = True,
                 dw_kernel_size: Union[int, List[int], Tuple[int]] = 7,
                 ):
        super().__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0

        self.kernel_size = to_3tuple(kernel_size)
        self.stride = to_3tuple(stride)
        self.padding = to_3tuple(padding)
        self.dilation = to_3tuple(dilation)
        self.groups = groups
        self.deformable_groups_per_groups = deformable_groups_per_groups
        self.offset_scale = offset_scale
        self.modulation_type = modulation_type

        kernel_sizes = multiply_integers(self.kernel_size)

        self.offset_field = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                padding=(dw_kernel_size - 1) // 2,
                groups=in_channels,
            ),
            group_norm(in_channels, num_groups=in_channels),
            nn.Conv3d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes * 3,
                kernel_size=1,
            )
        )

        self.attn_mask = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                padding=(dw_kernel_size - 1) // 2,
                groups=in_channels,
            ),
            group_norm(in_channels, num_groups=in_channels),
            nn.Conv3d(
                in_channels,
                groups * deformable_groups_per_groups * kernel_sizes,
                kernel_size=1,
            )
        )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self, mean: float = 0., std: float = 0.02):
        self.weight.data.normal_(mean=mean, std=std)
        if self.bias is not None:
            self.bias.data.normal_(mean=mean, std=std)

        nn.init.zeros_(self.offset_field[0].weight)
        nn.init.zeros_(self.offset_field[-1].weight)
        nn.init.zeros_(self.offset_field[0].bias)
        nn.init.zeros_(self.offset_field[-1].bias)

        nn.init.zeros_(self.attn_mask[0].weight)
        nn.init.zeros_(self.attn_mask[-1].weight)
        nn.init.zeros_(self.attn_mask[0].bias)
        nn.init.ones_(self.attn_mask[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset_field = self.offset_field(x) * self.offset_scale
        attn_mask = self.attn_mask(x)
        attn_mask = get_modulation_scalar(attn_mask, self.modulation_type, self.groups * self.deformable_groups_per_groups)

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
