# Pytorch-Implmentation-of-Deformable-Convolution-Nd   
Supporting N-dimensional deformable convolution    
This implementation is quite different from original implementation.   
   
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
- GPU: NVIDA 3070TI 8G in Windows10 / NVIDIA A5000 in Ubuntu
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
- [doc](https://github.com/seungjun-Park/Pytorch-Implmentation-of-Deformable-Convolution-Nd/edit/main/doc.md)

## Example  
```
import torch
import custom_op

device = 'cuda'

inp = torch.randn(2, 32, 64, 64).to(device)

# learnable parameter for deformable convolution.
weight = torch.randn(64, 32 // 2, 3, 3).to(device)
bias = torch.randn(64).to(device)

# may be created by nn.Conv, in this case 2d.
# the shape is [batch, groups * deformable_groups_per_group * kernel_h * kernel_w * (current deformable convolution dim), out_h, out_w], in this case.
offset_field = torch.randn(2, 16 * 9, 64, 64).to(device)

# the shape is [batch, groups * deformable_groups_per_group * kernel_h * kernel_w, out_h, out_w], in this case.
attn_mask = torch.randn(2, 1, 16 * 9, 64, 64).to(device)

# usecase. if you want to know what is each parameters meaning, please see doc.md
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
        2,        // deformable groups
        bias,
    )

print(output)
```
  
## Reference   
- [Pytorch official dispatch mechanism doc](https://pytorch.org/tutorials/advanced/dispatcher.html)  
- [Pytorch official/Dilated Convolution](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/NaiveDilatedConvolution.cpp)  
- [CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2)   
- [OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4)   
