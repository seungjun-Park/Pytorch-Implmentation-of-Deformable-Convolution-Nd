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

template<typename T, uint8_t size>
struct Array
{
	__host__ __device__ T operator[](uint8_t index) const
	{
		assert(index < size);
		return elements[index];
	}

	__host__ __device__ T& operator[](uint8_t index)
	{
		assert(index < size);
		return elements[index];
	}

	T elements[size];
};

template<uint8_t size>
using IntArray = Array<int64_t, size>;

template<typename T, uint8_t size>
Array<T, size> ArrayRef2Array(at::ArrayRef<T> arr)
{
	assert(arr.size() == size);
	Array<T, size> target;
	for (uint8_t i = 0; i < size; i++)
	{
		target[i] = arr[i];
	}

	return target;
}

template<uint8_t size>
IntArray<size> IntArrayRef2IntArray(at::IntArrayRef arr)
{
	assert(arr.size() == size);
	IntArray<size> target;
	for (uint8_t i = 0; i < size; i++)
	{
		target[i] = arr[i];
	}

	return target;
}

template<typename T, uint8_t size>
Array<T, size> vector2Array(std::vector<T>& vec)
{
	assert(vec.size() == size);
	Array<T, size> target;
	for (uint8_t i = 0; i < size; i++)
	{
		target[i] = vec[i];
	}

	return target;
}

template<typename T, uint8_t size>
__host__ __device__ T multiply_elements(const Array<T, size>& arr)
{
	T mul = 1;

	for (uint8_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return (T)mul;
}


template<uint8_t size>
__host__ __device__ int64_t multiply_integers(const IntArray<size>& arr)
{
	int64_t mul = 1;

	for (uint8_t i = 0; i < size; i++)
	{
		mul *= arr[i];
	}

	return (int64_t)mul;
}