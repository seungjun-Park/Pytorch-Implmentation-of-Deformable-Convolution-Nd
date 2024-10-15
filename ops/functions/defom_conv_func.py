from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Optional, List, Tuple, Union

from deform_conv import ext


class DeformConvNd(Function):
    @staticmethod
    @custom_fwd
    def forward(
            ctx,
            inps: torch.Tensor,
            weight: torch.Tensor,
            offset_field: torch.Tensor,
            attn_mask: torch.Tensor,
            kernel_size: Union[List[int], Tuple[int]],
            stride: Union[List[int], Tuple[int]] = 1,
            padding: Union[List[int], Tuple[int]] = 0,
            dilation: Union[List[int], Tuple[int]] = 1,
            groups: int = 1,
            bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        ctx.save_for_backward(inps, weight, offset_field, attn_mask, bias)

        return ext.deform_conv_forward(
            inps,
            weight,
            offset_field,
            attn_mask,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        inps, weight, offset_field, attn_mask, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_offset_field, grad_attn_mask, grad_bias = ext.deform_conv_backward(
            inps,
            weight,
            offset_field,
            attn_mask,
            grad_output.contiguous(),
            ctx.kernel_size,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
            bias
        )

        return grad_input, grad_weight, grad_offset_field, grad_attn_mask, None, None, None, None, None, grad_bias


_deform_conv_nd = DeformConvNd.apply
