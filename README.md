# Pytorch-implmentation-of-Deformable-Convolution-Nd   
Supporting N-dimensional deformable convolution    
This implementation is quite different from original implementation.   
   
please see original implementation [here](https://github.com/msracver/Deformable-ConvNets).   

## Papers   
[Deformable Convolution v1](https://arxiv.org/abs/1703.06211)   
[Deformable Convolution v2](https://arxiv.org/abs/1811.11168)   
[Deformable Convolution v3(InternImage)](https://arxiv.org/abs/2211.05778)   
[Deformable Convolution v4](https://arxiv.org/abs/2401.06197)   
   
## What's different from the original paper?   
Offset field:   
In general, the original paper implements offset field shape as (batch size, 2(dim), kernel height * kernel widht, out height, out width) in v1 and v2.  
In v3 and v4, the offset field shape is (batch size, 2(dim), groups * kernel height * kernel widht, out height, out width).   
   
In this implementation, however, the offset field shape is (batch size, 2(dim), groups * (in_channels / groups) * kernel height * kernel widht, out height, out width).  

Modulation scalar:   
Support sigmoid in v2, softmax in v3, non-bounded value range in v4, tanh(bounded value range to -1 ~ 1) in this version.  

## Requirements   
- Pytorch >= 2.1.0
- CudaToolkit >= 11.8
- Ninja (Optional)
   
## Test environments   
OS: Windows10 with MSVC / Linux(Ubuntu) with gcc   
  
## Build
```python
python setup.py build install
```
  
## Reference code   
[CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2)   
[OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4)   
