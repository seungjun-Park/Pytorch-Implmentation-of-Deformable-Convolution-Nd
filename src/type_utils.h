#pragma once

#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>
#include <type_traits>

// to support half precision cuda version

template<typename T>
struct type_mapper
{
	using type = T;
};

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
#include <cuda_fp16.h>
template<>
struct type_mapper<c10::Half>
{
	using type = half;
};

#endif

#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
#include <cuda_bf16.h>
template<>
struct type_mapper<c10::BFloat16>
{
	using type = nv_bfloat16;
};
#endif

template<typename T>
using mapped_type = typename type_mapper<T>::type;