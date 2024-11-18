# Pytorch-Implmentation-of-Deformable-Convolution-Nd   
Supporting N-dimensional deformable convolution     
   
Original implementation:    
[DCNV2](https://github.com/msracver/Deformable-ConvNets)  
[DCNV4](https://github.com/OpenGVLab/DCNv4)  

Model example: please see [here](https://github.com/seungjun-Park/Deformable-Edge-Detector)  

## Support  

- fp16. (cuda compute capability >= 7.0 && only gpu available, but when use fp16, it often occurs over/underflow problem. please use bfp16 instead.)  
- bfp16. (cuda compute capability >= 8.0)  
- torch.no_grad(). (enable gradient checkpointing)  
- torch.autocast(). (enable pytorch AMP system)  
- torch.autograd(). (just called function. do not need any additional implement.) 
- 1d ~ 3d implement. (if you want to use over 3d, you just add dimension which you wanted TORCH_LIBRARY_IMPL in deform_conv.cpp, deform_conv_cpu.cpp and deform_conv_cuda.cu)
- channels last memory format.  
  
## Papers   
- [Deformable Convolution v1](https://arxiv.org/abs/1703.06211)   
- [Deformable Convolution v2](https://arxiv.org/abs/1811.11168)   
- [Deformable Convolution v3(InternImage)](https://arxiv.org/abs/2211.05778)   
- [Deformable Convolution v4](https://arxiv.org/abs/2401.06197)   

## Requirements   
- Pytorch
- CudaToolkit
- Python
- Ninja (Optional for fast build)
   
## Test environments   
- OS: Windows10 with MSVC / Ubuntu 20.04.6 LTS with gcc
- GPU: NVIDA 3070TI 8G in Windows10 / NVIDIA A5000 24G in Ubuntu
- C++: std 17  
- C: std 14  
- Python: 3.10  
- Pytorch: 2.1.0  / 2.4.0
- CudaToolkit: 11.8  / 12.4
  
## Build
```python
(Optional) conda activate {your envirionment name}
python setup.py build install
```

## DOC  
- [doc](https://github.com/seungjun-Park/Pytorch-Implmentation-of-Deformable-Convolution-Nd/blob/main/doc.md)

## Example  
```
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
```
  
## Reference   
- [Pytorch official dispatch mechanism doc](https://pytorch.org/tutorials/advanced/dispatcher.html)  
- [Pytorch official/Dilated Convolution](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/NaiveDilatedConvolution.cpp)  
- [CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2)   
- [OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4)   
