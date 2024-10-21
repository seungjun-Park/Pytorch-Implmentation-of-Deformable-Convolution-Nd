#pragma once

#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// to support half precision cuda version

template<typename T>
struct type_mapper
{
	using type = T;
};

#ifdef WITH_CUDA
template<>
struct type_mapper<c10::Half>
{
	using type = __half;
};
#endif // WITH_CUDA

template<typename T>
using mapped_type = typename type_mapper<T>::type;