#pragma once

#include <torch/extension.h>
#include <interpolation.h>
#include <array_utils.h>
#include <type_traits>

//////////////////////      Implementation      ////////////////////////


// implementation of n-dimensional im2col.
template<typename T, int8_t dim, bool is_channels_last>
typename std::enable_if<(dim > IMPLEMENTED_DIM && !is_channels_last), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t grouped_channels,
    const IntArray<dim>& input_size,
    const IntArray<dim>& output_size,
    const IntArray<dim>& kernel_size,
    const IntArray<dim>& stride,
    const IntArray<dim>& padding,
    const IntArray<dim>& dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    const double_t offset_scale,
    const bool fix_center,
    T* data_col)
{
    const int64_t kernel_sizes = multiply_integers(kernel_size);
    const int64_t output_sizes = multiply_integers(output_size);
    const int64_t input_sizes = multiply_integers(input_size);

    int64_t current_output_size[dim];
    int64_t current_kernel_size[dim];

    for (int64_t g = 0; g < groups; g++)
    {
        for (int64_t ch = 0; ch < grouped_channels; ch++)
        {
            for (int64_t k = 0; k < kernel_sizes; k++)
            {
                for (int64_t col = 0; col < output_sizes; col++)
                {
                    int64_t d_g = ch * deformable_groups_per_groups / grouped_channels;

                    int64_t col_idx = ((g * grouped_channels + ch) * kernel_sizes + k) * output_sizes + col;
                    int64_t offset_field_idx = ((g * deformable_groups_per_groups + d_g) * (kernel_sizes - fix_center) + k) * dim * output_sizes + col;
                    int64_t attn_mask_idx = ((g * deformable_groups_per_groups + d_g) * kernel_sizes + k) * output_sizes + col;

                    Array<T, dim> coord;

                    // compute n-dimensional current kernel/output size
                    int64_t out_div = 1;
                    int64_t k_div = 1;

                    int64_t k_center = k / 2;
                    if (fix_center && k > k_center)
                    {
                        offset_field_idx -= dim * output_sizes;
                    }

                    for (int8_t i = dim - 1; i >= 0; i--)
                    {
                        current_output_size[i] = col / out_div % output_size[i];
                        current_kernel_size[i] = k / k_div % kernel_size[i];
                        out_div *= output_size[i];
                        k_div *= kernel_size[i];
                        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i];

                        if (!fix_center || k != k_center)
                        {
                            coord[i] += data_offset_field[offset_field_idx + i * output_sizes] * offset_scale;
                        }
                    }

                    T val = linear_interp_nd<T, dim, is_channels_last>(data_im + ch * input_sizes, coord, input_size, grouped_channels * groups);

                    data_col[col_idx] = val * data_attn_mask[attn_mask_idx];
                }
            }
        }

        data_im += grouped_channels * input_sizes;
    }
}

template<typename T, int8_t dim, bool is_channels_last>
typename std::enable_if<(dim == 1 && !is_channels_last), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t grouped_channels,
    const IntArray<1>& input_size,
    const IntArray<1>& output_size,
    const IntArray<1>& kernel_size,
    const IntArray<1>& stride,
    const IntArray<1>& padding,
    const IntArray<1>& dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    const double_t offset_scale,
    const bool fix_center,
    T* data_col) {

    for (int64_t g = 0; g < groups; g++)
    {
        for (int64_t ch = 0; ch < grouped_channels; ch++)
        {
            for (int64_t k = 0; k < kernel_size[0]; k++)
            {
                for (int64_t col = 0; col < output_size[0]; col++)
                {
                    int64_t d_g = ch * deformable_groups_per_groups / grouped_channels;

                    int64_t col_idx = ((g * grouped_channels + ch) * kernel_size[0] + k) * output_size[0] + col;
                    int64_t offset_field_idx = ((g * deformable_groups_per_groups + d_g) * (kernel_size[0] - fix_center) + k) * output_size[0] + col;
                    int64_t attn_mask_idx = ((g * deformable_groups_per_groups + d_g) * kernel_size[0] + k) * output_size[0] + col;

                    int64_t k_center = k / 2;
                    if (fix_center && k > k_center)
                    {
                        offset_field_idx -= output_size[0];
                    }

                    Array<T, 1> coord;
                    coord[0] = col * stride[0] - padding[0] + k * dilation[0];

                    if (!fix_center || k != k_center)
                    {
                        coord[0] += data_offset_field[offset_field_idx] * offset_scale;
                    }

                    T val = linear_interp_nd<T, 1, is_channels_last>(data_im + ch * input_size[0], coord, input_size, grouped_channels * groups);

                    data_col[col_idx] = val * data_attn_mask[attn_mask_idx];
                }
            }
        }
        data_im += grouped_channels * input_size[0];
    }
}

template<typename T, int8_t dim, bool is_channels_last>
typename std::enable_if<(dim == 2 && !is_channels_last), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t grouped_channels,
    const IntArray<2>& input_size,
    const IntArray<2>& output_size,
    const IntArray<2>& kernel_size,
    const IntArray<2>& stride,
    const IntArray<2>& padding,
    const IntArray<2>& dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    const double_t offset_scale,
    const bool fix_center,
    T* data_col) {

    int64_t input_sizes = multiply_integers<2>(input_size);
    int64_t output_sizes = multiply_integers<2>(output_size);
    int64_t kernel_sizes = multiply_integers<2>(kernel_size);

    for (int64_t g = 0; g < groups; g++)
    {
        for (int64_t ch = 0; ch < grouped_channels; ch++)
        {
            for (int64_t h_k = 0; h_k < kernel_size[0]; h_k++)
            {
                for (int64_t w_k = 0; w_k < kernel_size[1]; w_k++)
                {
                    for (int64_t h_col = 0; h_col < output_size[0]; h_col++)
                    {
                        for (int64_t w_col = 0; w_col < output_size[1]; w_col++)
                        {
                            int64_t d_g = ch * deformable_groups_per_groups / grouped_channels;

                            int64_t col_idx = ((((g * grouped_channels + ch) * kernel_size[0] + h_k) * kernel_size[1] + w_k) * output_size[0] + h_col) * output_size[1] + w_col;
                            int64_t offset_field_idx = ((g * deformable_groups_per_groups + d_g) * (kernel_sizes - fix_center) * 
                                2 * output_size[0] + h_col) * output_size[1] + w_col;
                            int64_t attn_mask_idx = ((((g * deformable_groups_per_groups + d_g) * kernel_size[0] + h_k) * kernel_size[1] + w_k) * 
                                output_size[0] + h_col) * output_size[1] + w_col;

                            int64_t k = h_k * kernel_size[1] + w_k;
                            int64_t k_center = k / 2;
                            offset_field_idx += k * 2 * output_sizes;
                            if (fix_center && k > k_center)
                            {
                                offset_field_idx -= 2 * output_sizes;
                            }

                            Array<T, 2> coord;
                            coord[0] = h_col * stride[0] - padding[0] + h_k * dilation[0];
                            coord[1] = w_col * stride[1] - padding[1] + w_k * dilation[1];

                            if (!fix_center || k != k_center)
                            {
                                coord[0] += data_offset_field[offset_field_idx] * offset_scale;
                                coord[1] += data_offset_field[offset_field_idx + output_sizes] * offset_scale;
                            }

                            T val = linear_interp_nd<T, 2, is_channels_last>(data_im + ch * input_sizes, coord, input_size, grouped_channels * groups);

                            data_col[col_idx] = val * data_attn_mask[attn_mask_idx];
                        }
                    }
                }
            }
        }
        data_im += grouped_channels * input_sizes;
    }
}

template<typename T, int8_t dim, bool is_channels_last>
typename std::enable_if<(dim == 3 && !is_channels_last), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t grouped_channels,
    const IntArray<3>& input_size,
    const IntArray<3>& output_size,
    const IntArray<3>& kernel_size,
    const IntArray<3>& stride,
    const IntArray<3>& padding,
    const IntArray<3>& dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    const double_t offset_scale,
    const bool fix_center,
    T* data_col) {

    int64_t input_sizes = multiply_integers<3>(input_size);
    int64_t output_sizes = multiply_integers<3>(output_size);
    int64_t kernel_sizes = multiply_integers<3>(kernel_size);

    for (int64_t g = 0; g < groups; g++)
    {
        for (int64_t ch = 0; ch < grouped_channels; ch++)
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
                                    int64_t d_g = ch * deformable_groups_per_groups / grouped_channels;

                                    int64_t col_idx = ((((((g * grouped_channels + ch) * kernel_size[0] + d_k) * kernel_size[1] + h_k) *
                                        kernel_size[2] + w_k) * output_size[0] + d_col) * output_size[1] + h_col) * output_size[2] + w_col;
                                    int64_t offset_field_idx = (((g * deformable_groups_per_groups + d_g) * (kernel_sizes - fix_center) * 3 * output_size[0] + 
                                        d_col) * output_size[1] + h_col) * output_size[2] + w_col;
                                    int64_t attn_mask_idx = ((((((g * deformable_groups_per_groups + d_g) * kernel_size[0] + d_k) * kernel_size[1] + h_k) * 
                                        kernel_size[2] + w_k) * output_size[0] + d_col) * output_size[1] + h_col) * output_size[2] + w_col;

                                    int64_t k = (d_k * kernel_size[1] + h_k) * kernel_size[2] + w_k;
                                    int64_t k_center = k / 2;
                                    offset_field_idx += k * 3 * output_sizes;
                                    if (fix_center && k > k_center)
                                    {
                                        offset_field_idx -= 3 * output_sizes;
                                    }

                                    Array<T, 3> coord;
                                    coord[0] = d_col * stride[0] - padding[0] + d_k * dilation[0];
                                    coord[1] = h_col * stride[1] - padding[1] + h_k * dilation[1];
                                    coord[2] = w_col * stride[2] - padding[2] + w_k * dilation[2];

                                    if (!fix_center || k != k_center)
                                    {
                                        coord[0] += data_offset_field[offset_field_idx] * offset_scale;
                                        coord[1] += data_offset_field[offset_field_idx + output_sizes] * offset_scale;
                                        coord[2] += data_offset_field[offset_field_idx + 2 * output_sizes] * offset_scale;
                                    }

                                    T val = linear_interp_nd<T, 3, is_channels_last>(data_im + ch * input_sizes, coord, input_size, grouped_channels * groups);

                                    data_col[col_idx] = val * data_attn_mask[attn_mask_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        data_im += grouped_channels * input_sizes;
    }
}

template<typename T, int8_t dim, bool is_channels_last>
typename std::enable_if<(dim > IMPLEMENTED_DIM && is_channels_last), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t grouped_channels,
    const IntArray<dim>& input_size,
    const IntArray<dim>& output_size,
    const IntArray<dim>& kernel_size,
    const IntArray<dim>& stride,
    const IntArray<dim>& padding,
    const IntArray<dim>& dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    const double_t offset_scale,
    const bool fix_center,
    T* data_col)
{
    const int64_t kernel_sizes = multiply_integers(kernel_size);
    const int64_t output_sizes = multiply_integers(output_size);
    const int64_t input_sizes = multiply_integers(input_size);

    int64_t current_output_size[dim];
    int64_t current_kernel_size[dim];

    for (int64_t col = 0; col < output_sizes; col++)
    {
        for (int64_t g = 0; g < groups; g++)
        {
            for (int64_t ch = 0; ch < grouped_channels; ch++)
            {
                for (int64_t k = 0; k < kernel_sizes; k++)
                {
                    int64_t d_g = ch * deformable_groups_per_groups / grouped_channels;

                    int64_t im_idx = g * grouped_channels + ch;
                    int64_t col_idx = ((col * groups + g) * grouped_channels + ch) * kernel_sizes + k;
                    int64_t offset_field_idx = (((col * groups + g) * deformable_groups_per_groups + d_g) * (kernel_sizes - fix_center) + k) * dim;
                    int64_t attn_mask_idx = ((col * groups + g) * deformable_groups_per_groups + d_g) * kernel_sizes + k;

                    Array<T, dim> coord;

                    int64_t k_center = k / 2;
                    if (fix_center && k > k_center)
                    {
                        offset_field_idx -= dim;
                    }

                    // compute n-dimensional current kernel/output size
                    int64_t out_div = 1;
                    int64_t k_div = 1;
                    for (int8_t i = dim - 1; i >= 0; i--)
                    {
                        current_output_size[i] = col / out_div % output_size[i];
                        current_kernel_size[i] = k / k_div % kernel_size[i];
                        out_div *= output_size[i];
                        k_div *= kernel_size[i];
                        coord[i] = current_output_size[i] * stride[i] - padding[i] + current_kernel_size[i] * dilation[i];
                    
                        if (!fix_center || k != k_center)
                        {
                            coord[i] += data_offset_field[offset_field_idx + i] * offset_scale;
                        }
                    }

                    T val = linear_interp_nd<T, dim, is_channels_last>(&data_im[im_idx], coord, input_size, grouped_channels * groups);

                    data_col[col_idx] = val * data_attn_mask[attn_mask_idx];
                }
            }
        }
    }
}

template<typename T, int8_t dim, bool is_channels_last>
typename std::enable_if<(dim == 1 && is_channels_last), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t grouped_channels,
    const IntArray<1>& input_size,
    const IntArray<1>& output_size,
    const IntArray<1>& kernel_size,
    const IntArray<1>& stride,
    const IntArray<1>& padding,
    const IntArray<1>& dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    const double_t offset_scale,
    const bool fix_center,
    T* data_col) {

    for (int64_t col = 0; col < output_size[0]; col++)
    {
        for (int64_t g = 0; g < groups; g++)
        {
            for (int64_t ch = 0; ch < grouped_channels; ch++)
            {
                for (int64_t k = 0; k < kernel_size[0]; k++)
                {
                    int64_t d_g = ch * deformable_groups_per_groups / grouped_channels;

                    int64_t im_idx = g * grouped_channels + ch;
                    int64_t col_idx = ((col * groups + g) * grouped_channels + ch) * kernel_size[0] + k;
                    int64_t offset_field_idx = ((col * groups + g) * deformable_groups_per_groups + d_g) * (kernel_size[0] - fix_center) + k;
                    int64_t attn_mask_idx = ((col * groups + g) * deformable_groups_per_groups + d_g) * kernel_size[0] + k;

                    int64_t k_center = k / 2;
                    if (fix_center && k > k_center)
                    {
                        offset_field_idx -= 1;
                    }

                    Array<T, 1> coord;
                    coord[0] = col * stride[0] - padding[0] + k * dilation[0];

                    if (!fix_center || k != k_center)
                    {
                        coord[0] += data_offset_field[offset_field_idx] * offset_scale;
                    }

                    T val = linear_interp_nd<T, 1, is_channels_last>(&data_im[im_idx], coord, input_size, grouped_channels * groups);

                    data_col[col_idx] = val * data_attn_mask[attn_mask_idx];
                }
            }
        }
    }
}

template<typename T, int8_t dim, bool is_channels_last>
typename std::enable_if<(dim == 2 && is_channels_last), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t grouped_channels,
    const IntArray<2>& input_size,
    const IntArray<2>& output_size,
    const IntArray<2>& kernel_size,
    const IntArray<2>& stride,
    const IntArray<2>& padding,
    const IntArray<2>& dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    const double_t offset_scale,
    const bool fix_center,
    T* data_col) {

    int64_t input_sizes = multiply_integers<2>(input_size);
    int64_t output_sizes = multiply_integers<2>(output_size);
    int64_t kernel_sizes = multiply_integers<2>(kernel_size);

    for (int64_t h_col = 0; h_col < output_size[0]; h_col++)
    {
        for (int64_t w_col = 0; w_col < output_size[1]; w_col++)
        {
            for (int64_t g = 0; g < groups; g++)
            {
                for (int64_t ch = 0; ch < grouped_channels; ch++)
                {
                    for (int64_t h_k = 0; h_k < kernel_size[0]; h_k++)
                    {
                        for (int64_t w_k = 0; w_k < kernel_size[1]; w_k++)
                        {
                            int64_t d_g = ch * deformable_groups_per_groups / grouped_channels;

                            int64_t im_idx = g * grouped_channels + ch;
                            int64_t col_idx = ((((h_col * output_size[1] + w_col) * groups + g) * grouped_channels + ch) * kernel_size[0] + h_k) * 
                                kernel_size[1] + w_k;
                            int64_t offset_field_idx = (((h_col * output_size[1] + w_col) * groups + g) * deformable_groups_per_groups + d_g) * 
                                (kernel_sizes - fix_center) * 2;
                            int64_t attn_mask_idx = ((((h_col * output_size[1] + w_col) * groups + g) * deformable_groups_per_groups + d_g) * 
                                kernel_size[0] + h_k) * kernel_size[1] + w_k;

                            int64_t k = h_k * kernel_size[1] + w_k;
                            int64_t k_center = k / 2;
                            offset_field_idx += k * 2;
                            if (fix_center && k > k_center)
                            {
                                offset_field_idx -= 2;
                            }

                            Array<T, 2> coord;
                            coord[0] = h_col * stride[0] - padding[0] + h_k * dilation[0];
                            coord[1] = w_col * stride[1] - padding[1] + w_k * dilation[1];

                            if (!fix_center || k != k_center)
                            {
                                coord[0] += data_offset_field[offset_field_idx] * offset_scale;
                                coord[1] += data_offset_field[offset_field_idx + 1] * offset_scale;
                            }

                            T val = linear_interp_nd<T, 2, is_channels_last>(&data_im[im_idx], coord, input_size, grouped_channels * groups);

                            data_col[col_idx] = val * data_attn_mask[attn_mask_idx];
                        }
                    }
                }
            }
        }
    }
}

template<typename T, int8_t dim, bool is_channels_last>
typename std::enable_if<(dim == 3 && is_channels_last), void>::type
im2col_nd_cpu(
    const T* data_im,
    const T* data_offset_field,
    const T* data_attn_mask,
    const int64_t grouped_channels,
    const IntArray<3>& input_size,
    const IntArray<3>& output_size,
    const IntArray<3>& kernel_size,
    const IntArray<3>& stride,
    const IntArray<3>& padding,
    const IntArray<3>& dilation,
    const int64_t groups,
    const int64_t deformable_groups_per_groups,
    const double_t offset_scale,
    const bool fix_center,
    T* data_col) {

    int64_t input_sizes = multiply_integers<3>(input_size);
    int64_t output_sizes = multiply_integers<3>(output_size);
    int64_t kernel_sizes = multiply_integers<3>(kernel_size);

    for (int64_t d_col = 0; d_col < output_size[0]; d_col++)
    {
        for (int64_t h_col = 0; h_col < output_size[1]; h_col++)
        {
            for (int64_t w_col = 0; w_col < output_size[2]; w_col++)
            {
                for (int64_t g = 0; g < groups; g++)
                {
                    for (int64_t ch = 0; ch < grouped_channels; ch++)
                    {
                        for (int64_t d_k = 0; d_k < kernel_size[0]; d_k++)
                        {
                            for (int64_t h_k = 0; h_k < kernel_size[1]; h_k++)
                            {
                                for (int64_t w_k = 0; w_k < kernel_size[2]; w_k++)
                                {
                                    int64_t d_g = ch * deformable_groups_per_groups / grouped_channels;

                                    int64_t im_idx = g * grouped_channels + ch;
                                    int64_t col_idx = (((((((d_col * output_size[1] + h_col) * output_size[2] + w_col) * groups + g) *
                                        grouped_channels + ch) * kernel_size[0] + d_k) * kernel_size[1] + h_k) * kernel_size[2] + w_k);
                                    int64_t offset_field_idx = ((((d_col * output_size[1] + h_col) * output_size[2] + w_col) *
                                        groups + g) * deformable_groups_per_groups + d_g) * (kernel_sizes - fix_center) * 3;
                                    int64_t attn_mask_idx = (((((((d_col * output_size[1] + h_col) * output_size[2] + w_col) *
                                        groups + g) * deformable_groups_per_groups + d_g) * kernel_size[0] + d_k) * kernel_size[1] + h_k) * kernel_size[2] + w_k);

                                    int64_t k = (d_k * kernel_size[1] + h_k) * kernel_size[2] + w_k;
                                    int64_t k_center = k / 2;
                                    offset_field_idx += k * 3;
                                    if (fix_center && k > k_center)
                                    {
                                        offset_field_idx -= 3;
                                    }

                                    Array<T, 3> coord;
                                    coord[0] = d_col * stride[0] - padding[0] + d_k * dilation[0] + data_offset_field[offset_field_idx];
                                    coord[1] = h_col * stride[1] - padding[1] + h_k * dilation[1] + data_offset_field[offset_field_idx + 1];
                                    coord[2] = w_col * stride[2] - padding[2] + w_k * dilation[2] + data_offset_field[offset_field_idx + 2];

                                    if (!fix_center || k != k_center)
                                    {
                                        coord[0] += data_offset_field[offset_field_idx] * offset_scale;
                                        coord[1] += data_offset_field[offset_field_idx + 1] * offset_scale;
                                        coord[2] += data_offset_field[offset_field_idx + 2] * offset_scale;
                                    }

                                    T val = linear_interp_nd<T, 3, is_channels_last>(&data_im[im_idx], coord, input_size, grouped_channels * groups);

                                    data_col[col_idx] = val * data_attn_mask[attn_mask_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

