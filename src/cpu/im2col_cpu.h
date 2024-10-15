#pragma once

#include <torch/extension.h>
#include <interpolation.h>
#include <utils.h>
#include <type_traits>

/////////////////////       Declaration     ///////////////////

template<typename T, int64_t dim>
typename std::enable_if<(dim > IMPLEMENTED_DIM), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t channels,
    const IntArray<dim>& input_size,
    const IntArray<dim>& output_size,
    const IntArray<dim>& kernel_size,
    const IntArray<dim>& stride,
    const IntArray<dim>& padding,
    const IntArray<dim>& dilation,
    const int64_t groups,
    T* data_col);

template<typename T, int64_t dim>
typename std::enable_if<(dim == 1), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t channels,
    const IntArray<1>& input_size,
    const IntArray<1>& output_size,
    const IntArray<1>& kernel_size,
    const IntArray<1>& stride,
    const IntArray<1>& padding,
    const IntArray<1>& dilation,
    const int64_t groups,
    T* data_col);

template<typename T, int64_t dim>
typename std::enable_if<(dim == 2), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t channels,
    const IntArray<2>& input_size,
    const IntArray<2>& output_size,
    const IntArray<2>& kernel_size,
    const IntArray<2>& stride,
    const IntArray<2>& padding,
    const IntArray<2>& dilation,
    const int64_t groups,
    T* data_col);

template<typename T, int64_t dim>
typename std::enable_if<(dim == 3), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t channels,
    const IntArray<3>& input_size,
    const IntArray<3>& output_size,
    const IntArray<3>& kernel_size,
    const IntArray<3>& stride,
    const IntArray<3>& padding,
    const IntArray<3>& dilation,
    const int64_t groups,
    T* data_col);


//////////////////////      Implementation      ////////////////////////


// implementation of n-dimensional im2col.
template<typename T, int64_t dim>
typename std::enable_if<(dim > IMPLEMENTED_DIM), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t channels,
    const IntArray<dim>& input_size,
    const IntArray<dim>& output_size,
    const IntArray<dim>& kernel_size,
    const IntArray<dim>& stride,
    const IntArray<dim>& padding,
    const IntArray<dim>& dilation,
    const int64_t groups,
    T* data_col)
{

    int64_t kernel_sizes = multiply_integers(kernel_size);
    int64_t output_sizes = multiply_integers(output_size);
    int64_t input_sizes = multiply_integers(input_size);

    // compute index for each dimension of offset field.
    int64_t base_offset_field_idx = groups * channels * kernel_sizes * output_sizes;

    IntArray<dim> current_output_size;
    IntArray<dim> current_kernel_size;

    for (int64_t group_idx = 0; group_idx < groups; group_idx++)
    {
        for (int64_t ch = 0; ch < channels; ch++)
        {
            for (int64_t k = 0; k < kernel_sizes; k++)
            {
                for (int64_t out = 0; out < output_sizes; out++)
                {
                    T val = 0;
                    FloatArray<dim> coord;
             
                    // compute n-dimensional current kernel/output size
                    int64_t out_div = 1;
                    int64_t k_div = 1;
                    for (int32_t i = dim - 1; i >= 0; i--)
                    {
                        current_output_size[i] = out / out_div % output_size[i];
                        current_kernel_size[i] = k / k_div % kernel_size[i];
                        out_div *= output_size[i];
                        k_div *= kernel_size[i];
                        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i] + *(data_offset_field + base_offset_field_idx * i);
                    }

                    val = linear_interp_nd<T, dim>(data_im + ch * input_sizes, coord, input_size);

                    *data_col = val * (*data_attn_mask);
                    data_col++;
                    data_attn_mask++;
                    data_offset_field++;
                }
            }
        }

        data_im += channels * input_sizes;
    }
}

template<typename T, int64_t dim>
typename std::enable_if<(dim == 1), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t channels,
    const IntArray<1>& input_size,
    const IntArray<1>& output_size,
    const IntArray<1>& kernel_size,
    const IntArray<1>& stride,
    const IntArray<1>& padding,
    const IntArray<1>& dilation,
    const int64_t groups,
    T* data_col) {

    const int64_t input_sizes = multiply_integers(input_size);
    const int64_t output_sizes = multiply_integers(output_size);
    const int64_t kernel_sizes = multiply_integers(kernel_size);

    for (int64_t group_idx = 0; group_idx < groups; group_idx++)
    {
        for (int64_t ch = 0; ch < channels; ch++)
        {
            for (int64_t k = 0; k < kernel_size[0]; k++)
            {
                for (int64_t col = 0; col < output_size[0]; col++)
                {
                    FloatArray<1> coord;
                    coord[0] = col * stride[0] - padding[0] + k * dilation[0] + (*data_offset_field);
                    T val = linear_interp_nd<T, dim>(data_im + ch * input_sizes, coord, input_size);

                    *data_col = val * (*data_attn_mask);

                    data_col++;
                    data_offset_field++;
                    data_attn_mask++;
                }
            }
        }
        data_im += channels * input_sizes;
    }
}

template<typename T, int64_t dim>
typename std::enable_if<(dim == 2), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t channels,
    const IntArray<2>& input_size,
    const IntArray<2>& output_size,
    const IntArray<2>& kernel_size,
    const IntArray<2>& stride,
    const IntArray<2>& padding,
    const IntArray<2>& dilation,
    const int64_t groups,
    T* data_col) {

    const int64_t input_sizes = multiply_integers(input_size);
    const int64_t output_sizes = multiply_integers(output_size);
    const int64_t kernel_sizes = multiply_integers(kernel_size);

    const T* data_offset_field_h = data_offset_field;
    const T* data_offset_field_w = (data_offset_field + groups * channels * kernel_sizes * output_sizes);

    for (int64_t g = 0; g < groups; g++)
    {
        for (int64_t ch = 0; ch < channels; ch++)
        {
            for (int64_t h_k = 0; h_k < kernel_size[0]; h_k++)
            {
                for (int64_t w_k = 0; w_k < kernel_size[1]; w_k++)
                {
                    for (int64_t h_col = 0; h_col < output_size[0]; h_col++)
                    {
                        for (int64_t w_col = 0; w_col < output_size[1]; w_col++)
                        {
                            FloatArray<2> coord;
                            coord[0] = h_col * stride[0] - padding[0] + h_k * dilation[0] + (*data_offset_field_h);
                            coord[1] = w_col * stride[1] - padding[1] + w_k * dilation[1] + (*data_offset_field_w);
                            T val = linear_interp_nd<T, 2>(data_im + ch * input_sizes, coord, input_size);

                            *data_col = val * (*data_attn_mask);
                            data_col++;
                            data_offset_field_h++;
                            data_offset_field_w++;
                            data_attn_mask++;
                        }
                    }
                }
            }
        }
        data_im += channels * input_sizes;
    }
}

template<typename T, int64_t dim>
typename std::enable_if<(dim == 3), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t channels,
    const IntArray<3>& input_size,
    const IntArray<3>& output_size,
    const IntArray<3>& kernel_size,
    const IntArray<3>& stride,
    const IntArray<3>& padding,
    const IntArray<3>& dilation,
    const int64_t groups,
    T* data_col) {

    const int64_t input_sizes = multiply_integers(input_size);
    const int64_t output_sizes = multiply_integers(output_size);
    const int64_t kernel_sizes = multiply_integers(kernel_size);

    const T* data_offset_field_d = data_offset_field;
    const T* data_offset_field_h = (data_offset_field + groups * channels * kernel_sizes * output_sizes);
    const T* data_offset_field_w = (data_offset_field + groups * channels * kernel_sizes * output_sizes * 2);

    for (int64_t group_idx = 0; group_idx < groups; group_idx++)
    {
        for (int64_t ch = 0; ch < channels; ch++)
        {
            for (int64_t d_k = 0; d_k < kernel_size[0]; d_k++)
            {
                for (int64_t h_k = 0; h_k < kernel_size[1]; h_k++)
                {
                    for (int64_t w_k = 0; w_k < kernel_size[2]; w_k++)
                    {
                        for (int64_t d_col = 0; d_col < output_size[0]; d_col++)
                        {
                            for (int64_t h_col = 0; h_col < output_size[1]; h_col++)
                            {
                                for (int64_t w_col = 0; w_col < output_size[2]; w_col++)
                                {
                                    float_t d_im = d_col * stride[0] - padding[0] + d_k * dilation[0] + (*data_offset_field_d);
                                    float_t h_im = h_col * stride[1] - padding[1] + h_k * dilation[1] + (*data_offset_field_h);
                                    float_t w_im = w_col * stride[2] - padding[2] + w_k * dilation[2] + (*data_offset_field_w);

                                    T val = 0.f;
                                    FloatArray<3> coord;
                                    coord[0] = d_col * stride[0] - padding[0] + d_k * dilation[0] + (*data_offset_field_d);
                                    coord[1] = h_col * stride[1] - padding[1] + h_k * dilation[1] + (*data_offset_field_h);
                                    coord[2] = w_col * stride[2] - padding[2] + w_k * dilation[2] + (*data_offset_field_w);
                                    val = linear_interp_nd<T, 3>(data_im + ch * input_sizes, coord, input_size);

                                    *data_col = val * (*data_attn_mask);
                                    data_col++;
                                    data_offset_field_d++;
                                    data_offset_field_h++;
                                    data_offset_field_w++;
                                    data_attn_mask++;
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

