#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <utils.h>
#include <type_traits>


/////////////       Implementation      ////////////////////


// n-dimensional linear interpolation.
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim > IMPLEMENTED_DIM), T>::type
linear_interp_nd(
    const T* data,
    const FloatArray<dim>& coord,
    const IntArray<dim>& data_size)
{
    // if idx < dim means coord_low, else coord_high.
    IntArray<dim * 2> coords;
    FloatArray<dim> ratios;

    for (int64_t i = 0; i < dim; i++)
    {
        coords[i] = floor(coord[i]);
        coords[i + dim] = coords[i] + 1;
        ratios[i] = coord[i] - coords[i];
    }

    constexpr int64_t num_points = 1 << dim;
    int64_t indice[dim] = { 0, };
    // 0 means low, 1 means high coord.
    int64_t elements[2] = { 0, 1 };

    T val = 0;

    // compute points with repeated permutation.
    for (int64_t idx = 0; idx < num_points; idx++)
    {
        int64_t div = 1;
        int64_t data_idx = 0;
        T weight = 1;
        bool is_valid_data = true;
        for (int i = dim - 1; i >= 0; i--)
        {
            int64_t current_coord = coords[elements[indice[i]] * dim + i];
            if (current_coord < 0 || current_coord >= data_size[i])
            {
                is_valid_data = false;
                break;
            }
            data_idx += div * current_coord;
            div *= data_size[i];
            weight *= elements[indice[i]] == 1 ? ratios[i] : 1.f - ratios[i];
        }

        if (is_valid_data)
        {
            val += weight * data[data_idx];
        }

        // compute next computation
        int pos = dim - 1;
        while (pos >= 0)
        {
            if (indice[pos] < 1)
            {
                indice[pos]++;
                break;
            }
            else
            {
                indice[pos] = 0;
                pos--;
            }
        }
    }

    return val;
}

// linear interpolation
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 1), T>::type
linear_interp_nd(
    const T* data,
    const FloatArray<1>& coord,
    const IntArray<1>& data_size) {

    /// data: [ element_length ] (1d).
    /// coord: 1d floating-point coordinate.
    /// data_size: size of data to each dimension. 

    int64_t low = floor(coord[0]);
    int64_t high = low + 1;

    // ratio
    float_t ratio = coord[0] - low;

    T v1 = 0, v2 = 0;

    if (low >= 0 && low < data_size[0])
        v1 = data[low];

    if (high >= 0 && high < data_size[0])
        v2 = data[high];

    // weight for each values
    T val = (1.f - ratio) * v1 + ratio * v2;

    return val;
}

// bilinear interpolation
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 2), T>::type
linear_interp_nd(
    const T* data,
    const FloatArray<2>& coord,
    const IntArray<2>& data_size) {

    /// data: [ height, width ] (2d).
    /// coord: 2d floating-point coordinate.
    /// data_size: size of data to each dimension. 

    int64_t h_low = floor(coord[0]);
    int64_t h_high = h_low + 1;

    int64_t w_low = floor(coord[1]);
    int64_t w_high = w_low + 1;

    // ratio
    float_t ratio_h = coord[0] - h_low;
    float_t ratio_w = coord[1] - w_low;
    
    T v11 = 0, v12 = 0, v21 = 0, v22 = 0;

    if (h_low >= 0 && h_low < data_size[0] && w_low >= 0 && w_low < data_size[1])
        v11 = data[h_low * data_size[1] + w_low];

    if (h_low >= 0 && h_low < data_size[0] && w_high >= 0 && w_high < data_size[1])
        v12 = data[h_low * data_size[1] + w_high];

    if (h_high >= 0 && h_high < data_size[0] && w_low >= 0 && w_low < data_size[1])
        v21 = data[h_high * data_size[1] + w_low];

    if (h_high >= 0 && h_high < data_size[0] && w_high >= 0 && w_high < data_size[1])
        v22 = data[h_high * data_size[1] + w_high];

    // weight
    float_t w11 = (1.f - ratio_h) * (1.f - ratio_w); 
    float_t w21 = ratio_h * (1.f - ratio_w);
    float_t w12 = (1.f - ratio_h) * ratio_w;
    float_t w22 = ratio_h * ratio_w;

    T val = w11 * v11 + w21 * v21 + w12 * v12 + w22 * v22;

    return val;
}

// trilinear interp
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 3), T>::type
linear_interp_nd(
    const T* data,
    const FloatArray<3>& coord,
    const IntArray<3>& data_size) {

    /// data: [ depth, height, width ] (3d).
    /// coord: 3d floating-point coordinate.
    /// data_size: size of data to each dimension. 

    // depth
    int64_t d_low = floor(coord[0]);
    int64_t d_high = d_low + 1;

    int64_t h_low = floor(coord[1]);
    int64_t h_high = h_low + 1;

    int64_t w_low = floor(coord[2]);
    int64_t w_high = w_low + 1;

    // ratio
    float_t ratio_d = coord[0] - d_low;
    float_t ratio_h = coord[1] - h_low;
    float_t ratio_w = coord[2] - w_low;

    T v111 = 0, v211 = 0, v121 = 0, v221 = 0, v112 = 0, v212 = 0, v122 = 0, v222 = 0;

    // if the coordinate in data range 
    if (d_low >= 0 && d_low < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_low >= 0 && w_low < data_size[2])
        v111 = data[d_low * data_size[1] * data_size[2] + h_low * data_size[2] + w_low];

    if (d_high >= 0 && d_high < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_low >= 0 && w_low < data_size[2])
        v211 = data[d_high * data_size[1] * data_size[2] + h_low * data_size[2] + w_low];

    if (d_low >= 0 && d_low < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_low >= 0 && w_low < data_size[2])
        v121 = data[d_low * data_size[1] * data_size[2] + h_high * data_size[2] + w_low];

    if (d_high >= 0 && d_high < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_low >= 0 && w_low < data_size[2])
        v221 = data[d_high * data_size[1] * data_size[2] + h_high * data_size[2] + w_low];

    if (d_low >= 0 && d_low < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_high >= 0 && w_high < data_size[2])
        v112 = data[d_low * data_size[1] * data_size[2] + h_low * data_size[2] + w_high];

    if (d_high >= 0 && d_high < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_high >= 0 && w_high < data_size[2])
        v212 = data[d_high * data_size[1] * data_size[2] + h_low * data_size[2] + w_high];

    if (d_low >= 0 && d_low < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_high >= 0 && w_high < data_size[2])
        v122 = data[d_low * data_size[1] * data_size[2] + h_high * data_size[2] + w_high];

    if (d_high >= 0 && d_high < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_high >= 0 && w_high < data_size[2])
        v222 = data[d_high * data_size[1] * data_size[2] + h_high * data_size[2] + w_high];

    // weight
    float_t w111 = (1.f - ratio_d) * (1.f - ratio_h) * (1.f - ratio_w);
    float_t w211 = ratio_d * (1.f - ratio_h) * (1.f - ratio_w);
    float_t w121 = (1.f - ratio_d) * ratio_h * (1.f - ratio_w);
    float_t w221 = ratio_d * ratio_h * (1.f - ratio_w);
    float_t w112 = (1.f - ratio_d) * (1.f - ratio_h) * ratio_w;
    float_t w212 = ratio_d * (1.f - ratio_h) * ratio_w;
    float_t w122 = (1.f - ratio_d) * ratio_h * ratio_w;
    float_t w222 = ratio_d * ratio_h * ratio_w;

    T val = w111 * v111 + w211 * v211 + w121 * v121 + w221 * v221 + w112 * v112 + w212 * v212 + w122 * v122 + w222 * v222;

    return val;
}

// n-dimensional linear interpolation grad.
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim > IMPLEMENTED_DIM), Array<T, dim>>::type
linear_interp_nd_grad(
    const T* data,
    const FloatArray<dim>& coord,
    const IntArray<dim>& data_size)
{
    Array<T, dim> grads;

    // if idx < dim means coord_low, else coord_high.
    IntArray<dim * 2> coords;
    FloatArray<dim> ratios;

    for (int64_t i = 0; i < dim; i++)
    {
        coords[i] = floor(coord[i]);
        coords[i + dim] = coords[i] + 1;
        ratios[i] = coord[i] - coords[i];
    }

    constexpr int64_t num_points = (1 << dim);
    int64_t indice[dim] = { 0, };
    // 0 means low, 1 means high coord.
    int64_t elements[2] = { 0, 1 };

    T points[num_points] = { 0, };
    T weights[num_points] = { 0, };

    T val = 0;

    // compute points with repeated permutation.
    for (int64_t idx = 0; idx < num_points; idx++)
    {
        int64_t div = 1;
        int64_t point_div = 1;
        int64_t data_idx = 0;
        int64_t point_idx = 0;
        T weight = 1;
        bool is_valid_data = true;
        for (int i = dim - 1; i >= 0; i--)
        {
            int64_t current_coord = coords[elements[indice[i]] * dim + i];
            if (current_coord < 0 || current_coord >= data_size[i])
            {
                is_valid_data = false;
                break;
            }
            data_idx += div * current_coord;
            point_idx += point_div * elements[indice[i]];
            div *= data_size[i];
            point_div *= 2;
            weight *= elements[indice[i]] == 1 ? ratios[i] : 1.f - ratios[i];
        }

        if (is_valid_data)
        {
            points[point_idx] = data[data_idx];
            weights[point_idx] = weight;
        }

        // compute next computation
        int pos = dim - 1;
        while (pos >= 0)
        {
            if (indice[pos] < 1)
            {
                indice[pos]++;
                break;
            }
            else
            {
                indice[pos] = 0;
                pos--;
            }
        }
    }

    for (int64_t i = 0; i < dim; i++)
    {
        T grad = 0;
        for (int64_t point_idx = 0; point_idx < num_points; point_idx++)
        {
            int64_t current_point_shape[dim];
            int64_t point_div = 1;
            for (int j = dim - 1; j >= 0; j--)
            {
                current_point_shape[j] = point_idx / point_div % 2;
                point_div *= 2;
            }
            grad += ((current_point_shape[i] == 1) ? 1.f / (ratios[i] + 1e-5) : -1.f / (1.f - ratios[i] + 1e-5)) * weights[point_idx] * points[point_idx];
        }
        grads[i] = grad;
    }

    return grads;
}

// gradient of linear interpolation
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 1), Array<T, 1>>::type
linear_interp_nd_grad(
    const T* data,
    const FloatArray<1>& coord,
    const IntArray<1>& data_size) {

    /// data: [ element_length ] (1d).
    /// coord: 1d floating-point coordinate.
    /// data_size: size of data to each dimension. 

    int64_t low = floor(coord[0]);
    int64_t high = low + 1;

    // ratio
    float_t ratio = coord[0] - low;

    T v1 = 0, v2 = 0;

    if (low >= 0 && low < data_size[0])
        v1 = data[low];

    if (high >= 0 && high < data_size[0])
        v2 = data[high];

    // weight for each values
    T grad = v2 - v1;

    Array<T, 1> grads;
    grads[0] = grad;

    return grads;
}

// grdient of bilinear interpolation
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 2), Array<T, 2>>::type
linear_interp_nd_grad(
    const T* data,
    const FloatArray<2>& coord,
    const IntArray<2>& data_size) {

    /// data: [ height, width ] (2d).
    /// coord: 2d floating-point coordinate.
    /// data_size: size of data to each dimension. 

    int64_t h_low = floor(coord[0]);
    int64_t h_high = h_low + 1;

    int64_t w_low = floor(coord[1]);
    int64_t w_high = w_low + 1;

    // ratio
    float_t ratio_h = coord[0] - h_low;
    float_t ratio_w = coord[1] - w_low;

    T v11 = 0, v12 = 0, v21 = 0, v22 = 0;

    if (h_low >= 0 && h_low < data_size[0] && w_low >= 0 && w_low < data_size[1])
        v11 = data[h_low * data_size[1] + w_low];

    if (h_low >= 0 && h_low < data_size[0] && w_high >= 0 && w_high < data_size[1])
        v12 = data[h_low * data_size[1] + w_high];

    if (h_high >= 0 && h_high < data_size[0] && w_low >= 0 && w_low < data_size[1])
        v21 = data[h_high * data_size[1] + w_low];

    if (h_high >= 0 && h_high < data_size[0] && w_high >= 0 && w_high < data_size[1])
        v22 = data[h_high * data_size[1] + w_high];

    // gradient of height and width
    T grad_h = (v21 - v11) * (1 - ratio_w) + (v22 - v12) * ratio_w;
    T grad_w = (v12 - v11) * (1 - ratio_h) + (v22 - v21) * ratio_h;

    Array<T, 2> grads;
    grads[0] = grad_h;
    grads[1] = grad_w;

    return grads;
}

// trilinear interp
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 3), Array<T, 3>>::type
linear_interp_nd_grad(
    const T* data,
    const FloatArray<3>& coord,
    const IntArray<3>& data_size) {

    /// data: [ depth, height, width ] (3d).
    /// coord: 3d floating-point coordinate.
    /// data_size: size of data to each dimension. 

    // depth
    int64_t d_low = floor(coord[0]);
    int64_t d_high = d_low + 1;

    int64_t h_low = floor(coord[1]);
    int64_t h_high = h_low + 1;

    int64_t w_low = floor(coord[2]);
    int64_t w_high = w_low + 1;

    // ratio
    float_t ratio_d = coord[0] - d_low;
    float_t ratio_h = coord[1] - h_low;
    float_t ratio_w = coord[2] - w_low;

    T v111 = 0, v211 = 0, v121 = 0, v221 = 0, v112 = 0, v212 = 0, v122 = 0, v222 = 0;

    // if the coordinate in data range 
    if (d_low >= 0 && d_low < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_low >= 0 && w_low < data_size[2])
        v111 = data[d_low * data_size[1] * data_size[2] + h_low * data_size[2] + w_low];

    if (d_high >= 0 && d_high < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_low >= 0 && w_low < data_size[2])
        v211 = data[d_high * data_size[1] * data_size[2] + h_low * data_size[2] + w_low];

    if (d_low >= 0 && d_low < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_low >= 0 && w_low < data_size[2])
        v121 = data[d_low * data_size[1] * data_size[2] + h_high * data_size[2] + w_low];

    if (d_high >= 0 && d_high < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_low >= 0 && w_low < data_size[2])
        v221 = data[d_high * data_size[1] * data_size[2] + h_high * data_size[2] + w_low];

    if (d_low >= 0 && d_low < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_high >= 0 && w_high < data_size[2])
        v112 = data[d_low * data_size[1] * data_size[2] + h_low * data_size[2] + w_high];

    if (d_high >= 0 && d_high < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_high >= 0 && w_high < data_size[2])
        v212 = data[d_high * data_size[1] * data_size[2] + h_low * data_size[2] + w_high];

    if (d_low >= 0 && d_low < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_high >= 0 && w_high < data_size[2])
        v122 = data[d_low * data_size[1] * data_size[2] + h_high * data_size[2] + w_high];

    if (d_high >= 0 && d_high < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_high >= 0 && w_high < data_size[2])
        v222 = data[d_high * data_size[1] * data_size[2] + h_high * data_size[2] + w_high];

    T grad_d = (1 - ratio_h) * (1 - ratio_w) * (v211 - v111) + ratio_h * (1 - ratio_w) * (v221 - v121) + (1 - ratio_h) * ratio_w * (v212 - v112) + ratio_h * ratio_w * (v222 - v122);
    T grad_h = (1 - ratio_d) * (1 - ratio_w) * (v121 - v111) + ratio_d * (1 - ratio_w) * (v221 - v211) + (1 - ratio_d) * ratio_w * (v122 - v112) + ratio_d * ratio_w * (v222 - v212);
    T grad_w = (1 - ratio_d) * (1 - ratio_h) * (v112 - v111) + ratio_d * (1 - ratio_h) * (v212 = v211) + (1 - ratio_d) * ratio_h * (v122 - v121) + ratio_d * ratio_h * (v222 - v221);

    Array<T, 3> grads;
    grads[0] = grad_d;
    grads[1] = grad_h;
    grads[2] = grad_w;

    return grads;
}

// n-dimensional linear interpolation weight.
template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim > IMPLEMENTED_DIM), void>::type
linear_interp_nd_weight(
    const T col,
    const T attn_mask,
    const FloatArray<dim>& coord,
    const IntArray<dim>& data_size,
    T* data_grad)
{
    // if idx < dim means coord_low, else coord_high.
    IntArray<dim * 2> coords;
    FloatArray<dim> ratios;

    for (int64_t i = 0; i < dim; i++)
    {
        coords[i] = floor(coord[i]);
        coords[i + dim] = coords[i] + 1;
        ratios[i] = coord[i] - coords[i];
    }

    constexpr int64_t num_points = 1 << dim;
    int64_t indice[dim] = { 0, };
    // 0 means low, 1 means high coord.
    int64_t elements[2] = { 0, 1 };

    // compute points with repeated permutation.
    for (int64_t idx = 0; idx < num_points; idx++)
    {
        int64_t div = 1;
        int64_t data_idx = 0;
        T weight = 1;
        bool is_valid_data = true;
        for (int i = dim - 1; i >= 0; i--)
        {
            int64_t current_coord = coords[elements[indice[i]] * dim + i];
            if (current_coord < 0 || current_coord >= data_size[i])
            {
                is_valid_data = false;
                break;
            }
            data_idx += div * current_coord;
            div *= data_size[i];
            weight *= elements[indice[i]] == 1 ? ratios[i] : 1.f - ratios[i];
        }

        if (is_valid_data)
        {
#ifdef __CUDA_ARCH__
            atomicAdd(&data_grad[data_idx], weight * col * attn_mask);
#else
            data_grad[data_idx] += weight * col * attn_mask;
#endif // __CUDA_ARCH__
        }

        // compute next computation
        int pos = dim - 1;
        while (pos >= 0)
        {
            if (indice[pos] < 1)
            {
                indice[pos]++;
                break;
            }
            else
            {
                indice[pos] = 0;
                pos--;
            }
        }
    }
}

template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 1), void>::type
linear_interp_nd_weight(
    const T col,
    const T attn_mask,
    const FloatArray<1>& coord,
    const IntArray<1>& data_size,
    T* data_grad)
{
    /// col: specific value of columns which grad_output @ weight^T.
    /// attn_mask: specific value of attn_mask data.
    /// coord: 1d floating-point coordinate.
    /// data_size: size of data to each dimension. 
    /// data_grad: [ elements ] (1d)

    int64_t low = floor(coord[0]);
    int64_t high = low + 1;

    // ratio
    float_t ratio = coord[0] - low;

    T w1 = 1.f - ratio;
    T w2 = ratio;
#ifdef __CUDA_ARCH__
    if (low >= 0 && low < data_size[0])
        atomicAdd(&data_grad[low], w1 * attn_mask * col);

    if (high >= 0 && high < data_size[0])
        atomicAdd(&data_grad[high], w2 * attn_mask * col);
#else
    if (low >= 0 && low < data_size[0])
        data_grad[low] += w1 * attn_mask * col;

    if (high >= 0 && high < data_size[0])
        data_grad[high] += w2 * attn_mask * col;
#endif // __CUDA_ARCH__
}

template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 2), void>::type
linear_interp_nd_weight(
    const T col,
    const T attn_mask,
    const FloatArray<2>& coord,
    const IntArray<2>& data_size,
    T* data_grad)
{
    /// col: specific value of columns which grad_output @ weight^T.
    /// attn_mask: specific value of attn_mask data.
    /// coord: 2d floating-point coordinate.
    /// data_size: size of data to each dimension. 
    /// data_grad: [ hegith, width ] (2d)

    int64_t h_low = floor(coord[0]);
    int64_t h_high = h_low + 1;

    int64_t w_low = floor(coord[1]);
    int64_t w_high = w_low + 1;

    // ratio
    float_t ratio_h = coord[0] - h_low;
    float_t ratio_w = coord[1] - w_low;

    // weight
    float_t w11 = (1 - ratio_h) * (1 - ratio_w);
    float_t w21 = ratio_h * (1 - ratio_w);
    float_t w12 = (1 - ratio_h) * ratio_w;
    float_t w22 = ratio_h * ratio_w;

#ifdef __CUDA_ARCH__
    if (h_low >= 0 && h_low < data_size[0] && w_low >= 0 && w_low < data_size[1])
    {
        atomicAdd(&data_grad[h_low * data_size[1] + w_low], attn_mask * col * w11);
    }

    if (h_high >= 0 && h_high < data_size[0] && w_low >= 0 && w_low < data_size[1])
    {
        atomicAdd(&data_grad[h_high * data_size[1] + w_low], attn_mask * col * w21);
    }

    if (h_low >= 0 && h_low < data_size[0] && w_high >= 0 && w_high < data_size[1])
    {
        atomicAdd(&data_grad[h_low * data_size[1] + w_high], attn_mask * col * w12);
    }

    if (h_high >= 0 && h_high < data_size[0] && w_high >= 0 && w_high < data_size[1])
    {
        atomicAdd(&data_grad[h_high * data_size[1] + w_high], attn_mask * col * w22);
    }
#else
    if (h_low >= 0 && h_low < data_size[0] && w_low >= 0 && w_low < data_size[1])
    {
        data_grad[h_low * data_size[1] + w_low] += attn_mask * col * w11;
    }

    if (h_high >= 0 && h_high < data_size[0] && w_low >= 0 && w_low < data_size[1])
    {
        data_grad[h_high * data_size[1] + w_low] += attn_mask * col * w21;
    }

    if (h_low >= 0 && h_low < data_size[0] && w_high >= 0 && w_high < data_size[1])
    {
        data_grad[h_low * data_size[1] + w_high] += attn_mask * col * w12;
    }

    if (h_high >= 0 && h_high < data_size[0] && w_high >= 0 && w_high < data_size[1])
    {
        data_grad[h_high * data_size[1] + w_high] += attn_mask * col * w22;
    }
#endif // __CUDA_ARCH__
}

template<typename T, int64_t dim>
__host__ __device__
typename std::enable_if<(dim == 3), void>::type
linear_interp_nd_weight(
    const T col,
    const T attn_mask,
    const FloatArray<3>& coord,
    const IntArray<3>& data_size,
    T* data_grad)
{
    /// col: specific value of columns which grad_output @ weight^T.
    /// attn_mask: specific value of attn_mask data.
    /// coord: 3d floating-point coordinate.
    /// data_size: size of data to each dimension. 
    /// data_grad: [ depth, hegith, width ] (3d)

    // depth
    int64_t d_low = floor(coord[0]);
    int64_t d_high = d_low + 1;

    int64_t h_low = floor(coord[1]);
    int64_t h_high = h_low + 1;

    int64_t w_low = floor(coord[2]);
    int64_t w_high = w_low + 1;

    // ratio
    float_t ratio_d = coord[0] - d_low;
    float_t ratio_h = coord[1] - h_low;
    float_t ratio_w = coord[2] - w_low;

    // weight
    float_t w111 = (1.f - ratio_d) * (1.f - ratio_h) * (1.f - ratio_w);
    float_t w211 = ratio_d * (1.f - ratio_h) * (1.f - ratio_w);
    float_t w121 = (1.f - ratio_d) * ratio_h * (1.f - ratio_w);
    float_t w221 = ratio_d * ratio_h * (1.f - ratio_w);
    float_t w112 = (1.f - ratio_d) * (1.f - ratio_h) * ratio_w;
    float_t w212 = ratio_d * (1.f - ratio_h) * ratio_w;
    float_t w122 = (1.f - ratio_d) * ratio_h * ratio_w;
    float_t w222 = ratio_d * ratio_h * ratio_w;

#ifdef __CUDA_ARCH__
    if (d_low >= 0 && d_low < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_low >= 0 && w_low < data_size[2])
        atomicAdd(&data_grad[d_low * data_size[1] * data_size[2] + h_low * data_size[2] + w_low], w111 * attn_mask * col);

    if (d_high >= 0 && d_high < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_low >= 0 && w_low < data_size[2])
        atomicAdd(&data_grad[d_high * data_size[1] * data_size[2] + h_low * data_size[2] + w_low], w211 * attn_mask * col);

    if (d_low >= 0 && d_low < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_low >= 0 && w_low < data_size[2])
        atomicAdd(&data_grad[d_low * data_size[1] * data_size[2] + h_high * data_size[2] + w_low], w121 * attn_mask * col);

    if (d_high >= 0 && d_high < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_low >= 0 && w_low < data_size[2])
        atomicAdd(&data_grad[d_high * data_size[1] * data_size[2] + h_high * data_size[2] + w_low], w221 * attn_mask * col);

    if (d_low >= 0 && d_low < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_high >= 0 && w_high < data_size[2])
        atomicAdd(&data_grad[d_low * data_size[1] * data_size[2] + h_low * data_size[2] + w_high], w112 * attn_mask * col);

    if (d_high >= 0 && d_high < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_high >= 0 && w_high < data_size[2])
        atomicAdd(&data_grad[d_high * data_size[1] * data_size[2] + h_low * data_size[2] + w_high], w212 * attn_mask * col);

    if (d_low >= 0 && d_low < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_high >= 0 && w_high < data_size[2])
        atomicAdd(&data_grad[d_low * data_size[1] * data_size[2] + h_high * data_size[2] + w_high], w122 * attn_mask * col);

    if (d_high >= 0 && d_high < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_high >= 0 && w_high < data_size[2])
        atomicAdd(&data_grad[d_high * data_size[1] * data_size[2] + h_high * data_size[2] + w_high], w222 * attn_mask * col);
#else
    if (d_low >= 0 && d_low < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_low >= 0 && w_low < data_size[2])
        data_grad[d_low * data_size[1] * data_size[2] + h_low * data_size[2] + w_low] += w111 * attn_mask * col;

    if (d_high >= 0 && d_high < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_low >= 0 && w_low < data_size[2])
        data_grad[d_high * data_size[1] * data_size[2] + h_low * data_size[2] + w_low] += w211 * attn_mask * col;

    if (d_low >= 0 && d_low < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_low >= 0 && w_low < data_size[2])
        data_grad[d_low * data_size[1] * data_size[2] + h_high * data_size[2] + w_low] += w121 * attn_mask * col;

    if (d_high >= 0 && d_high < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_low >= 0 && w_low < data_size[2])
        data_grad[d_high * data_size[1] * data_size[2] + h_high * data_size[2] + w_low] += w221 * attn_mask * col;

    if (d_low >= 0 && d_low < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_high >= 0 && w_high < data_size[2])
        data_grad[d_low * data_size[1] * data_size[2] + h_low * data_size[2] + w_high] += w112 * attn_mask * col;

    if (d_high >= 0 && d_high < data_size[0] && h_low >= 0 && h_low < data_size[1] && w_high >= 0 && w_high < data_size[2])
        data_grad[d_high * data_size[1] * data_size[2] + h_low * data_size[2] + w_high] += w212 * attn_mask * col;

    if (d_low >= 0 && d_low < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_high >= 0 && w_high < data_size[2])
        data_grad[d_low * data_size[1] * data_size[2] + h_high * data_size[2] + w_high] += w122 * attn_mask * col;

    if (d_high >= 0 && d_high < data_size[0] && h_high >= 0 && h_high < data_size[1] && w_high >= 0 && w_high < data_size[2])
        data_grad[d_high * data_size[1] * data_size[2] + h_high * data_size[2] + w_high] += w222 * attn_mask * col;
#endif // __CUDA_ARCH__
}