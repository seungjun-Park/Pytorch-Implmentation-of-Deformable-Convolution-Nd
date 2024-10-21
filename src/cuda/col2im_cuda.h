#pragma once

#include <torch/extension.h>
#include <sm_60_atomic_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <interpolation.h>
#include <array_utils.h>
#include <type_utils.h>
#include <type_traits>


///////////////////     Implementation      ///////////////////////

template<typename T, int8_t dim>
__global__
typename std::enable_if<(dim > 0), void>::type
col2im_nd_cuda(
    const T* data_im,
    const T* data_col,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int32_t sub_batch,
    const int32_t channels,
    const IntArray<dim> input_size,
    const IntArray<dim> output_size,
    const IntArray<dim> kernel_size,
    const IntArray<dim> stride,
    const IntArray<dim> padding,
    const IntArray<dim> dilation,
    const int32_t groups,
    const int32_t offset_field_channels_per_groups,
    mapped_type<T>* data_grad_im,
    mapped_type<T>* data_grad_offset_field,
    mapped_type<T>* data_grad_attn_mask
) {
    int32_t input_sizes = multiply_integers(input_size);
    int32_t output_sizes = multiply_integers(output_size);
    int32_t kernel_sizes = multiply_integers(kernel_size);

    int64_t num_col = sub_batch * groups * channels * kernel_sizes * output_sizes;
    int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_col)
    {
        return;
    }

    int32_t col = idx % output_sizes;
    int32_t k = idx / output_sizes % kernel_sizes;
    int32_t ch = idx / (kernel_sizes * output_sizes) % channels;
    int32_t g = idx / (channels * kernel_sizes * output_sizes) % groups;
    int32_t batch_idx = idx / (groups * channels * kernel_sizes * output_sizes) % sub_batch;

    Array<T, dim> coord;

    int32_t current_kernel_size[dim];
    int32_t current_output_size[dim];

    int32_t k_div = 1;
    int32_t out_div = 1;

    int64_t col_idx = (((g * channels + ch) * kernel_sizes + k) * sub_batch + batch_idx) * output_sizes + col;
    int64_t im_idx = ((batch_idx * groups + g) * channels + ch) * input_sizes;
    int64_t offset_field_idx = (((batch_idx * dim * groups + g) * offset_field_channels_per_groups + (ch * offset_field_channels_per_groups / channels)) * kernel_sizes + k) * output_sizes + col;
    int64_t base_offset_field_idx = groups * offset_field_channels_per_groups * kernel_sizes * output_sizes;
    int64_t attn_mask_idx = (((batch_idx * groups + g) * offset_field_channels_per_groups + (ch * offset_field_channels_per_groups / channels)) * kernel_sizes + k) * output_sizes + col;

    data_im += im_idx;
    data_grad_im += im_idx;
    
    data_col += col_idx;

    // compute current kernel size, output size and coord.
    for (int8_t i = dim - 1; i >= 0; i--)
    {
        current_kernel_size[i] = k / k_div % kernel_size[i];
        current_output_size[i] = col / out_div % output_size[i];
        k_div *= kernel_size[i];
        out_div *= output_size[i];

        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + (data_offset_field + base_offset_field_idx * i)[offset_field_idx];
    }

    T val = linear_interp_nd<T, dim>(data_im, coord, input_size);
    atomicAdd(&data_grad_attn_mask[attn_mask_idx], (*data_col) * val);

    Array<T, dim> grad_coord = linear_interp_nd_grad<T, dim>(data_im, coord, input_size);

    for (int8_t i = dim - 1; i >= 0; i--)
    {
        atomicAdd(&((data_grad_offset_field + i * base_offset_field_idx)[offset_field_idx]), (*data_col) * grad_coord[i] * data_attn_mask[attn_mask_idx]);
    }

    linear_interp_nd_weight<T, dim>(*data_col, data_attn_mask[attn_mask_idx], coord, input_size, data_grad_im);
}