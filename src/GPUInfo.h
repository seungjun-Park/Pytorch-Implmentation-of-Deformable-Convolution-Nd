#pragma once
#include <cuda_runtime.h>

class GPUInfo final
{
public:
	GPUInfo(bool printInfo = false);
	~GPUInfo();

	int GetDeviceCount() const;
	const cudaDeviceProp* GetDeviceProps() const;

private:
	void checkDevice(bool printInfo = false);

private:
	int deviceCount = -1;
	cudaDeviceProp* deviceProps = nullptr;
};