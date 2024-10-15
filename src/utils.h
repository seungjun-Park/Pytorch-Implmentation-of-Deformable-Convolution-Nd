#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <vector>

// replace to at::IntArrayRef in device code.
// it's usage is equal to at::IntArrayRef in deform conv cpu version.

#ifndef WITH_CUDA
#define WITH_CUDA
#endif // !WITH_CUDA


#ifndef IMPLEMENTED_DIM
#define IMPLEMENTED_DIM 3
#endif // !IMPLEMENTED_DIM

template<typename T, int64_t size>
struct Array
{
	__host__ __device__ T operator[](int64_t index) const
	{
		assert(index < size);
		return elements[index];
	}

	__host__ __device__ T& operator[](int64_t index)
	{
		assert(index < size);
		return elements[index];
	}

	T elements[size];
};

template<int64_t size>
using IntArray = Array<int64_t, size>;

template<int64_t size>
using FloatArray = Array<float_t, size>;

template<typename T, int64_t size>
Array<T, size> ArrayRef2Array(at::ArrayRef<T> arr)
{
	assert(arr.size() == size);
	Array<T, size> target;
	for (size_t i = 0; i < length; i++)
	{
		target[i] = arr[i];
	}

	return target;
}

template<int64_t size>
IntArray<size> IntArrayRef2IntArray(at::IntArrayRef arr)
{
	assert(arr.size() == size);
	IntArray<size> target;
	for (size_t i = 0; i < size; i++)
	{
		target[i] = arr[i];
	}

	return target;
}

template<typename T, int64_t size>
Array<T, size> vector2Array(std::vector<int64_t>& vec)
{
	assert(vec.size() == size);
	Array<T, size> target;
	for (size_t i = 0; i < size; i++)
	{
		target[i] = vec[i];
	}

	return target;
}

template<typename T, int64_t size>
__host__ __device__ T multiply_elements(const Array<T, size>& arr)
{
	T mul = 1;

	for (size_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return (T)mul;
}

template<int64_t size>
__host__ __device__ int64_t multiply_integers(const Array<int64_t, size>& arr)
{
	int64_t mul = 1;

	for (size_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return mul;
}