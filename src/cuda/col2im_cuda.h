#pragma once

#include <torch/extension.h>
#include <sm_60_atomic_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <interpolation.h>
#include <utils.h>
#include <type_traits>


///////////////////     Implementation      ///////////////////////

template<typename T, int64_t dim>
__global__
typename std::enable_if<(dim > 0), void>::type
col2im_nd_cuda(
    const T* data_im,
    const T* data_col,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<dim> input_size,
    const IntArray<dim> output_size,
    const IntArray<dim> kernel_size,
    const IntArray<dim> stride,
    const IntArray<dim> padding,
    const IntArray<dim> dilation,
    const int64_t groups,
    T* data_grad_im,
    T* data_grad_offset_field,
    T* data_grad_attn_mask
) {
    int64_t input_sizes = multiply_integers(input_size);
    int64_t output_sizes = multiply_integers(output_size);
    int64_t kernel_sizes = multiply_integers(kernel_size);

    int64_t num_col = sub_batch * groups * channels * kernel_sizes * output_sizes;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_col)
    {
        return;
    }

    int64_t col = idx % output_sizes;
    int64_t k = idx / output_sizes % kernel_sizes;
    int64_t ch = idx / (kernel_sizes * output_sizes) % channels;
    int64_t group_idx = idx / (channels * kernel_sizes * output_sizes) % groups;
    int64_t batch_idx = idx / (groups * channels * kernel_sizes * output_sizes) % sub_batch;

    T val = 0.f;
    FloatArray<dim> coord;

    int64_t current_kernel_size[dim];
    int64_t current_output_size[dim];

    int64_t k_div = 1;
    int64_t out_div = 1;

    int64_t col_idx = (((group_idx * channels + ch) * kernel_sizes + k) * sub_batch + batch_idx) * output_sizes + col;
    int64_t im_idx = ((batch_idx * groups + group_idx) * channels + ch) * input_sizes;
    int64_t offset_field_idx = (((batch_idx * dim * groups + group_idx) * channels + ch) * kernel_sizes + k) * output_sizes + col;
    int64_t additional_offset_field_idx = groups * channels * kernel_sizes * output_sizes;
    int64_t attn_mask_idx = (((batch_idx * groups + group_idx) * channels + ch) * kernel_sizes + k) * output_sizes + col;

    data_im += im_idx;
    data_grad_im += im_idx;
    
    data_col += col_idx;

    data_offset_field += offset_field_idx;
    data_grad_offset_field += offset_field_idx;

    data_attn_mask += attn_mask_idx;
    data_grad_attn_mask += attn_mask_idx;

    // compute current kernel size, output size and coord.
    for (int i = dim - 1; i >= 0; i--)
    {
        current_kernel_size[i] = k / k_div % kernel_size[i];
        current_output_size[i] = col / out_div % output_size[i];
        k_div *= kernel_size[i];
        out_div *= output_size[i];

        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + *(data_offset_field + additional_offset_field_idx * i);
    }

    val = linear_interp_nd<T, dim>(data_im, coord, input_size);
    *data_grad_attn_mask = (*data_col) * val;

    Array<T, dim> grad_coord = linear_interp_nd_grad<T, dim>(data_im, coord, input_size);

    for (int32_t i = dim - 1; i >= 0; i--)
    {
        *(data_grad_offset_field + i * additional_offset_field_idx) = (*data_col) * grad_coord[i] * (*data_attn_mask);
    }

    linear_interp_nd_weight<T, dim>(*data_col, *data_attn_mask, coord, input_size, data_grad_im);
}