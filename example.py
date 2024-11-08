import torch
import torch.nn as nn
from deform_conv import DeformConv2d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

inputs = torch.randn(2, 384, 64, 64).to(device)

deform_conv_layer = DeformConv2d(
    in_channels=384,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=8,
    deformable_groups_per_groups=2,
    bias=True
).to(device)

with torch.autocast(device_type=device, dtype=dtype):
    output = deform_conv_layer(inputs)

print(output)
"""
result is 
 1 2 3 4 ..... each values
 ...    ....     ........
 
 device='cuda: 0', dtype=torch.bfloat16,
 grad_fn=<CppNode<class DeformConvNdFunction<2>>>)
 DeformConvNdFunction located at src/deform_conv.cpp
"""
