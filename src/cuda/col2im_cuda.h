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
    const int64_t sub_batch,
    const int64_t channels,
    const IntArray<dim> input_size,
    const IntArray<dim> output_size,
    const IntArray<dim> kernel_size,
    const IntArray<dim> stride,
    const IntArray<dim> padding,
    const IntArray<dim> dilation,
    const int64_t groups,
    const int64_t deformable_groups,
    mapped_type<T>* data_grad_im,
    mapped_type<T>* data_grad_offset_field,
    mapped_type<T>* data_grad_attn_mask) {

    int64_t output_sizes = multiply_integers<dim>(output_size);
    int64_t kernel_sizes = multiply_integers<dim>(kernel_size);

    int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t numel = groups * kernel_sizes * channels * sub_batch * output_sizes;

    if (idx >= numel)
    {
        return;
    }

    int64_t col = idx % output_sizes;
    int64_t b = idx / output_sizes % sub_batch;
    int64_t k = idx / (sub_batch * output_sizes) % kernel_sizes;
    int64_t ch = idx / (kernel_sizes * sub_batch * output_sizes) % channels;
    int64_t g = idx / (channels * kernel_sizes * sub_batch * output_sizes) % groups;

    int64_t deformable_channels = channels / deformable_groups;

    data_im += ((b * groups + g) * channels + ch) * multiply_integers<dim>(input_size);
    data_col += (((g * channels + ch) * kernel_sizes + k) * sub_batch + b) * output_sizes + col;
    data_offset_field += (((b * groups + g) * deformable_groups + ch / deformable_channels) * kernel_sizes + k) * dim * output_sizes + col;
    data_attn_mask += (((b * groups + g) * deformable_groups + ch / deformable_channels) * kernel_sizes + k) * output_sizes + col;
    
    data_grad_im += ((b * groups + g) * channels + ch) * multiply_integers<dim>(input_size);
    data_grad_offset_field += (((b * groups + g) * deformable_groups + ch / deformable_channels) * kernel_sizes + k) * dim * output_sizes + col;
    data_grad_attn_mask += (((b * groups + g) * deformable_groups + ch / deformable_channels) * kernel_sizes + k) * output_sizes + col;

    int64_t current_output_size[dim];
    int64_t current_kernel_size[dim];
    Array<T, dim> coord;

    int64_t out_div = 1;
    int64_t k_div = 1;
    // compute current kernel size, output size and coord.
    for (int8_t i = dim - 1; i >= 0; i--)
    {
        current_kernel_size[i] = k / k_div % kernel_size[i];
        current_output_size[i] = col / out_div % output_size[i];
        out_div *= output_size[i];
        k_div *= kernel_size[i];
        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + *(data_offset_field + i * output_sizes);
    }

    T val = linear_interp_nd<T, dim>(data_im, coord, input_size);
    atomicAdd(data_grad_attn_mask, (mapped_type<T>)((*data_col) * val));

    Array<T, dim> grad_coord = linear_interp_nd_grad<T, dim>(data_im, coord, input_size);

    for (int8_t i = 0; i < dim; i++)
    {
        atomicAdd(data_grad_offset_field + i * output_sizes, (mapped_type<T>)((*data_col) * grad_coord[i] * (*data_attn_mask)));
    }

    linear_interp_nd_weight<T, dim>(*data_col, *data_attn_mask, coord, input_size, data_grad_im);
}