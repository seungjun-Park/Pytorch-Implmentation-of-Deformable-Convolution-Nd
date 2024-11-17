#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <array_utils.h>
#include <interpolation.h>
#include <type_traits>

// implementation of n-dimensional im2col.
// unlike cpu version, cuda version was implemented only n-dimensional.
// because n-dimension specific version has same overhead to n-dimensional verison.

template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim > 0 && !is_channels_last), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t grouped_channels,
    const IntArray<dim> input_size,
    const IntArray<dim> output_size,
    const IntArray<dim> kernel_size,
    const IntArray<dim> stride,
    const IntArray<dim> padding,
    const IntArray<dim> dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    T* data_col) {

    int64_t kernel_sizes = multiply_integers<dim>(kernel_size);
    int64_t input_sizes = multiply_integers<dim>(input_size);
    int64_t output_sizes = multiply_integers<dim>(output_size);

    int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t numel = groups * sub_batch * output_sizes * grouped_channels * kernel_sizes;

    if (idx >= numel)
    {
        return;
    }

    int64_t col = idx % output_sizes;
    int64_t b = idx / (output_sizes) % sub_batch;
    int64_t k = idx / (sub_batch * output_sizes) % kernel_sizes;
    int64_t ch = idx / (kernel_sizes * sub_batch * output_sizes) % grouped_channels;
    int64_t g = idx / (grouped_channels * kernel_sizes * sub_batch * output_sizes) % groups;

    int64_t d_g = (ch * deformable_groups_per_groups) / grouped_channels;

    data_im += ((b * groups + g) * grouped_channels + ch) * input_sizes;
    data_col += (((g * grouped_channels + ch) * kernel_sizes + k) * sub_batch + b) * output_sizes + col;
    data_offset_field += (((b * groups + g) * deformable_groups_per_groups + d_g) * kernel_sizes + k) * dim * output_sizes + col;
    data_attn_mask += (((b * groups + g) * deformable_groups_per_groups + d_g) * kernel_sizes + k) * output_sizes + col;

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
        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + data_offset_field[i * output_sizes];
    }

    T val = linear_interp_nd<T, dim, is_channels_last>(data_im, coord, input_size, grouped_channels * groups);

    *data_col = val * (*data_attn_mask);
}

template<typename T, int8_t dim, bool is_channels_last>
__global__
typename std::enable_if <(dim > 0 && is_channels_last), void>::type
im2col_nd_cuda(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t sub_batch,
    const int64_t grouped_channels,
    const IntArray<dim> input_size,
    const IntArray<dim> output_size,
    const IntArray<dim> kernel_size,
    const IntArray<dim> stride,
    const IntArray<dim> padding,
    const IntArray<dim> dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    T* data_col) {

    int64_t kernel_sizes = multiply_integers<dim>(kernel_size);
    int64_t input_sizes = multiply_integers<dim>(input_size);
    int64_t output_sizes = multiply_integers<dim>(output_size);

    int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t numel = groups * sub_batch * output_sizes * grouped_channels * kernel_sizes;

    if (idx >= numel)
    {
        return;
    }

    int64_t k = idx % kernel_sizes;
    int64_t ch = idx / (kernel_sizes) % grouped_channels;
    int64_t g = idx / (grouped_channels * kernel_sizes) % groups;
    int64_t col = idx / (groups * grouped_channels * kernel_sizes) % output_sizes;
    int64_t b = idx / (output_sizes * groups * grouped_channels * kernel_sizes) % sub_batch;

    int64_t d_g = (ch * deformable_groups_per_groups) / grouped_channels;

    data_im += ((b * input_sizes * groups + g) * grouped_channels + ch);
    data_col += (((b * output_sizes + col) * groups + g) * grouped_channels + ch) * kernel_sizes + k;
    data_offset_field += ((((b * output_sizes + col) * groups + g) * deformable_groups_per_groups + d_g) * kernel_sizes + k) * dim;
    data_attn_mask += (((b * output_sizes + col) * groups + g) * deformable_groups_per_groups + d_g) * kernel_sizes + k;

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
        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + data_offset_field[i];
    }

    T val = linear_interp_nd<T, dim, is_channels_last>(data_im, coord, input_size, grouped_channels * groups);

    *data_col = val * (*data_attn_mask);
}