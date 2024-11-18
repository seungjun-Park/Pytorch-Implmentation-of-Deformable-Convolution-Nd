# Deformable Convolution DOC  

## Parameters
- in_channels: channels of input  
- out_channels: channels of output  
- kernel_size: kernel_size like in convolution  
- stride: stride like in convolution
- padding: padding like in convolution
- dilation: dilation like in convolution
- groups: groups like in convolution
- deformable_groups_per_groups: when in_channels != out_channels and use deformable_group_channels, the groups may be in_channels // deforamble_group_channels != out_channels // deformable_group_channels.
  to solve the problem, groups = gcd(in_channels // deformable_groups, out_channels // deforamble_groups) and deformable_groups_per_groups = (in_channels // deformable_group_channels) // groups, default value is 1. 
- offset_scale: how many affect offset.
- modulation_type: to decide modulation calculate method. sigmoid in DCNv2, softmax in DCNv3, none in DCNv4
- bias: bias like in convolution
- dw_kernel_size: the depth-wise kernel_size to use offset_field and attn_mask

## Role of each member function and variable  
- self.offset_field: calculated by depth-wise seperable convolution
- self.attn_mask: calculated by depth-wise seperable convolution
- self._reset_parameters: to work regular convolution when initial training process. offset field = 0, attn_mask = 1 in initial value.
