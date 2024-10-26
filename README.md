# Pytorch-Implmentation-of-Deformable-Convolution-Nd   
Supporting N-dimensional deformable convolution    
This implementation is quite different from original implementation.   
   
please see original implementation  
[DCNV2](https://github.com/msracver/Deformable-ConvNets)  
[DCNV4](https://github.com/OpenGVLab/DCNv4)  

## Release note  

- Support fp16. (cuda compute capability >= 7.0 && only gpu available, but when use fp16, it often occurs over/underflow problem. please use bfp16 instead.)  
- Support bfp16. (cuda compute capability >= 8.0)  
- Support torch.no_grad().
- Support torch.autocast().
- Support torch.autograd().  
- Support 1d ~ 3d implement. (if you want to use over 3d, you just add dimension which you wanted TORCH_LIBRARY_IMPL in deform_conv.cpp, deform_conv_cpu.cpp and deform_conv_cuda.cu)       
- Support offset_feild_channel_per_groups params. (if this value is 1, then it is equal to paper version)  

## To do  

- Reduce GPU occupancy.  

## Papers   
- [Deformable Convolution v1](https://arxiv.org/abs/1703.06211)   
- [Deformable Convolution v2](https://arxiv.org/abs/1811.11168)   
- [Deformable Convolution v3(InternImage)](https://arxiv.org/abs/2211.05778)   
- [Deformable Convolution v4](https://arxiv.org/abs/2401.06197)   

## Mechanism   

## Requirements   
- Pytorch
- CudaToolkit >= 11.0 (Cuda compute capability >= 8.0 to support f16 and bf16)
- Python
- Ninja (Optional for fast build)
   
## Test environments   
- OS: Windows10 with MSVC / Linux(Ubuntu) with gcc  
- C++: std 17  
- C: std 14  
- Python: 3.10  
- Pytorch: 2.1.0  
- CudaToolkit: 11.8  
  
## Build
```python
conda activate {your envirionment name}
python setup.py build install
```

## Example  
```
import torch
import custom_op

device = 'cuda'

inp = torch.randn(2, 32, 64, 64).to(device)
weight = torch.randn(64, 32 // 2, 3, 3).to(device)
bias = torch.randn(64).to(device)
offset_field = torch.randn(2, 2, 16 * 9, 64, 64).to(device)
attn_mask = torch.randn(2, 1, 16 * 9, 64, 64).to(device)

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = torch.ops.custom_op.deform_conv2d(
        inp,
        weight,
        offset_field,
        attn_mask,
        (3, 3),   // kernel_size
        (1, 1),   // stride
        (1, 1),   // padding
        (1, 1),   // dilation
        2,        // groups
        8,        // offset_field_channels_per_groups
        bias,
    )

print(output)
```
  
## Reference   
- [Pytorch official dispatch mechanism doc](https://pytorch.org/tutorials/advanced/dispatcher.html)  
- [Pytorch official/Dilated Convolution](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/NaiveDilatedConvolution.cpp)  
- [CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2)   
- [OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4)   
