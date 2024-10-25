#pragma once

#include <torch/extension.h>
#include <interpolation.h>
#include <array_utils.h>
#include <type_traits>

//////////////////////      Implementation      ////////////////////////


// implementation of n-dimensional im2col.
template<typename T, int8_t dim>
typename std::enable_if<(dim > IMPLEMENTED_DIM), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t channels,
    const IntArray<dim>& input_size,
    const IntArray<dim>& output_size,
    const IntArray<dim>& kernel_size,
    const IntArray<dim>& stride,
    const IntArray<dim>& padding,
    const IntArray<dim>& dilation,
    const int32_t groups,
    const int32_t offset_field_channels_per_groups,
    T* data_col)
{

    const int32_t kernel_sizes = multiply_integers(kernel_size);
    const int32_t output_sizes = multiply_integers(output_size);
    const int32_t input_sizes = multiply_integers(input_size);

    // compute index for each dimension of offset field.
    const int32_t base_offset_field_idx = groups * offset_field_channels_per_groups * kernel_sizes * output_sizes;
    const int32_t offset_field_channels = channels / offset_field_channels_per_groups;

    IntArray<dim> current_output_size;
    IntArray<dim> current_kernel_size;

    for (int32_t g = 0; g < groups; g++)
    {
        for (int32_t ch = 0; ch < channels; ch++)
        {
            for (int32_t k = 0; k < kernel_sizes; k++)
            {
                for (int32_t col = 0; col < output_sizes; col++)
                {
                    Array<T, dim> coord;
                    
                    int64_t idx = ((g * offset_field_channels_per_groups + (ch * offset_field_channels_per_groups / channels)) *
                        kernel_sizes + k) * output_sizes + col;

                    // compute n-dimensional current kernel/output size
                    int32_t out_div = 1;
                    int32_t k_div = 1;
                    for (int8_t i = dim - 1; i >= 0; i--)
                    {
                        current_output_size[i] = col / out_div % output_size[i];
                        current_kernel_size[i] = k / k_div % kernel_size[i];
                        out_div *= output_size[i];
                        k_div *= kernel_size[i];
                        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + (data_offset_field + base_offset_field_idx * i)[idx];
                    }

                    T val = linear_interp_nd<T, dim>(data_im + ch * input_sizes, coord, input_size);

                    *data_col = val * data_attn_mask[idx];
                    data_col++;
                }
            }
        }

        data_im += channels * input_sizes;
    }
}

template<typename T, int8_t dim>
typename std::enable_if<(dim == 1), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t channels,
    const IntArray<1>& input_size,
    const IntArray<1>& output_size,
    const IntArray<1>& kernel_size,
    const IntArray<1>& stride,
    const IntArray<1>& padding,
    const IntArray<1>& dilation,
    const int32_t groups,
    const int32_t offset_field_channels_per_groups,
    T* data_col) {

    const int32_t input_sizes = multiply_integers(input_size);
    const int32_t output_sizes = multiply_integers(output_size);
    const int32_t kernel_sizes = multiply_integers(kernel_size);

    for (int32_t g = 0; g < groups; g++)
    {
        for (int32_t ch = 0; ch < channels; ch++)
        {
            for (int32_t k = 0; k < kernel_size[0]; k++)
            {
                for (int32_t col = 0; col < output_size[0]; col++)
                {
                    int64_t idx = ((g * offset_field_channels_per_groups + (ch * offset_field_channels_per_groups / channels)) *
                        kernel_sizes + k) * output_sizes + col;

                    Array<T, 1> coord;
                    coord[0] = col * stride[0] - padding[0] + k * dilation[0] + data_offset_field[idx];
                    T val = linear_interp_nd<T, dim>(data_im + ch * input_sizes, coord, input_size);

                    *data_col = val * data_attn_mask[idx];

                    data_col++;
                }
            }
        }
        data_im += channels * input_sizes;
    }
}

template<typename T, int8_t dim>
typename std::enable_if<(dim == 2), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t channels,
    const IntArray<2>& input_size,
    const IntArray<2>& output_size,
    const IntArray<2>& kernel_size,
    const IntArray<2>& stride,
    const IntArray<2>& padding,
    const IntArray<2>& dilation,
    const int32_t groups,
    const int32_t offset_field_channels_per_groups,
    T* data_col) {

    const int32_t input_sizes = multiply_integers(input_size);
    const int32_t output_sizes = multiply_integers(output_size);
    const int32_t kernel_sizes = multiply_integers(kernel_size);

    const T* data_offset_field_h = data_offset_field;
    const T* data_offset_field_w = (data_offset_field + groups * offset_field_channels_per_groups * kernel_sizes * output_sizes);

    for (int32_t g = 0; g < groups; g++)
    {
        for (int32_t ch = 0; ch < channels; ch++)
        {
            for (int32_t h_k = 0; h_k < kernel_size[0]; h_k++)
            {
                for (int32_t w_k = 0; w_k < kernel_size[1]; w_k++)
                {
                    for (int32_t h_col = 0; h_col < output_size[0]; h_col++)
                    {
                        for (int32_t w_col = 0; w_col < output_size[1]; w_col++)
                        {
                            int64_t idx = ((((g * offset_field_channels_per_groups + ((ch * offset_field_channels_per_groups) / channels)) *
                                kernel_size[0] + h_k) * kernel_size[1] + w_k) * output_size[0] + h_col) * output_size[1] + w_col;

                            Array<T, 2> coord;
                            coord[0] = h_col * stride[0] - padding[0] + h_k * dilation[0] + data_offset_field_h[idx];
                            coord[1] = w_col * stride[1] - padding[1] + w_k * dilation[1] + data_offset_field_w[idx];
                            T val = linear_interp_nd<T, 2>(data_im + ch * input_sizes, coord, input_size);

                            *data_col = val * data_attn_mask[idx];
                            data_col++;
                        }
                    }
                }
            }
        }
        data_im += channels * input_sizes;
    }
}

template<typename T, int8_t dim>
typename std::enable_if<(dim == 3), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t channels,
    const IntArray<3>& input_size,
    const IntArray<3>& output_size,
    const IntArray<3>& kernel_size,
    const IntArray<3>& stride,
    const IntArray<3>& padding,
    const IntArray<3>& dilation,
    const int32_t groups,
    const int32_t offset_field_channels_per_groups,
    T* data_col) {

    const int32_t input_sizes = multiply_integers(input_size);
    const int32_t output_sizes = multiply_integers(output_size);
    const int32_t kernel_sizes = multiply_integers(kernel_size);

    const T* data_offset_field_d = data_offset_field;
    const T* data_offset_field_h = (data_offset_field + groups * offset_field_channels_per_groups * kernel_sizes * output_sizes);
    const T* data_offset_field_w = (data_offset_field + groups * offset_field_channels_per_groups * kernel_sizes * output_sizes * 2);

    for (int32_t g = 0; g < groups; g++)
    {
        for (int32_t ch = 0; ch < channels; ch++)
        {
            for (int32_t d_k = 0; d_k < kernel_size[0]; d_k++)
            {
                for (int32_t h_k = 0; h_k < kernel_size[1]; h_k++)
                {
                    for (int32_t w_k = 0; w_k < kernel_size[2]; w_k++)
                    {
                        for (int32_t d_col = 0; d_col < output_size[0]; d_col++)
                        {
                            for (int32_t h_col = 0; h_col < output_size[1]; h_col++)
                            {
                                for (int32_t w_col = 0; w_col < output_size[2]; w_col++)
                                {
                                    int64_t idx = (((((g * offset_field_channels_per_groups + (ch * offset_field_channels_per_groups / channels) *
                                        kernel_size[0] + d_k) * kernel_size[1] + h_k) * kernel_size[2] + w_k) * output_size[0] + d_col) * output_size[1] + h_col) * output_size[2] + w_col;
                                    
                                    Array<T, 3> coord;
                                    coord[0] = d_col * stride[0] - padding[0] + d_k * dilation[0] + data_offset_field_d[idx];
                                    coord[1] = h_col * stride[1] - padding[1] + h_k * dilation[1] + data_offset_field_h[idx];
                                    coord[2] = w_col * stride[2] - padding[2] + w_k * dilation[2] + data_offset_field_w[idx];
                                    T val = linear_interp_nd<T, 3>(data_im + ch * input_sizes, coord, input_size);

                                    *data_col = val * data_attn_mask[idx];
                                    data_col++;
                                }
                            }
                        }
                    }
                }
            }
        }
        // add group index size.
        data_im += channels * input_sizes;
    }
}


