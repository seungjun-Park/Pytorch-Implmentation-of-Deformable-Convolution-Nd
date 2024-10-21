#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>

// replace to at::IntArrayRef in device code.
// it's usage is equal to at::IntArrayRef in deform conv cpu version.

#ifndef IMPLEMENTED_DIM
#define IMPLEMENTED_DIM 3
#endif // !IMPLEMENTED_DIM

template<typename T, uint32_t size>
struct Array
{
	__host__ __device__ T operator[](uint32_t index) const
	{
		assert(index < size);
		return elements[index];
	}

	__host__ __device__ T& operator[](uint32_t index)
	{
		assert(index < size);
		return elements[index];
	}

	T elements[size];
};

template<uint32_t size>
using IntArray = Array<int32_t, size>;

template<uint32_t size>
using LongIntArray = Array<int64_t, size>;

template<uint32_t size>
using FloatArray = Array<float_t, size>;

template<typename T, int64_t size>
Array<T, size> ArrayRef2Array(at::ArrayRef<T> arr)
{
	assert(arr.size() == size);
	Array<T, size> target;
	for (size_t i = 0; i < size; i++)
	{
		target[i] = arr[i];
	}

	return target;
}

template<uint32_t size>
IntArray<size> IntArrayRef2IntArray(at::IntArrayRef arr)
{
	assert(arr.size() == size);
	IntArray<size> target;
	for (size_t i = 0; i < size; i++)
	{
		target[i] = static_cast<int32_t>(arr[i]);
	}

	return target;
}

template<typename T, uint32_t size>
Array<T, size> vector2Array(std::vector<T>& vec)
{
	assert(vec.size() == size);
	Array<T, size> target;
	for (size_t i = 0; i < size; i++)
	{
		target[i] = vec[i];
	}

	return target;
}

template<typename T, uint32_t size>
__host__ __device__ T multiply_elements(const Array<T, size>& arr)
{
	T mul = 1;

	for (size_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return (T)mul;
}

template<uint32_t size>
__host__ __device__ uint32_t multiply_integers(const IntArray<size>& arr)
{
	int32_t mul = 1;

	for (uint32_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return mul;
}
