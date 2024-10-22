# Pytorch-Implmentation-of-Deformable-Convolution-Nd   
Supporting N-dimensional deformable convolution    
This implementation is quite different from original implementation.   
   
please see original implementation [here](https://github.com/msracver/Deformable-ConvNets).   

## Release note  

- Support fp16. (cuda compute capability >= 7.0 && only gpu available)
- Support bfp16. (cuda compute capability >= 8.0)  
- Support torch.no_grad().  
- Support 1d ~ Nd implement. (maximum dimension is 127).       
- Support offset_feild_channel_per_groups params. (if this value is 1, then it is equal to paper version)  

## Papers   
- [Deformable Convolution v1](https://arxiv.org/abs/1703.06211)   
- [Deformable Convolution v2](https://arxiv.org/abs/1811.11168)   
- [Deformable Convolution v3(InternImage)](https://arxiv.org/abs/2211.05778)   
- [Deformable Convolution v4](https://arxiv.org/abs/2401.06197)   

## Mechanism   

## Requirements   
- Pytorch
- CudaToolkit >= 11.0 (Cuda capability >= 8.0 to support f16 and bf16)
- Python
- Ninja (Optional for fast build)
   
## Test environments   
OS: Windows10 with MSVC / Linux(Ubuntu) with gcc  
C++: std 17  
C: std 14  
Python: 3.10  
Pytorch: 2.1.0  
CudaToolkit: 11.8  
  
## Build
```python
python setup.py build install
```

## Example  
```
```
  
## Reference code   
[Pytorch official/Dilated Convolution](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/NaiveDilatedConvolution.cpp)  
[CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2)   
[OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4)   
