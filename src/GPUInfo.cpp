#include <assert.h>
#include <string>
#include <iostream>
#include "GPUInfo.h"

GPUInfo::GPUInfo(bool printInfo)
{
    checkDevice(printInfo);
}

GPUInfo::~GPUInfo()
{
    if (this->deviceProps != nullptr)
    {
        delete[] this->deviceProps;
    }
}

int GPUInfo::GetDeviceCount() const
{
	return this->deviceCount;
}

const cudaDeviceProp* GPUInfo::GetDeviceProps() const
{
	return this->deviceProps;
}

void GPUInfo::checkDevice(bool printInfo)
{
    cudaError_t cudaStatus = cudaGetDeviceCount(&this->deviceCount);
    assert(cudaStatus == cudaError_t::cudaSuccess, std::stirng("Error fetching device count: ") + cudaGetErrorString(cudaStatus));

    deviceProps = new cudaDeviceProp[this->deviceCount];

    for (int deviceNum = 0; deviceNum < this->deviceCount; ++deviceNum) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceNum);
        deviceProps[deviceNum] = deviceProp;

        if (printInfo)
        {
            std::cout << "Device " << deviceNum << ": " << deviceProp.name << std::endl;
            std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
            std::cout << "Clock rate: " << deviceProp.clockRate << std::endl;
            std::cout << "Device copy overlap: ";
            if (deviceProp.deviceOverlap)
            {
                std::cout << "Enabled" << std::endl;
            }
            else
            {
                std::cout << "Disabled" << std::endl;
            }
            std::cout << "Kernel exceition timeout: ";
            if (deviceProp.kernelExecTimeoutEnabled)
            {
                std::cout << "Enabled" << std::endl;
            }
            else
            {
                std::cout << "Disabled" << std::endl;
            }

            std::cout << "Global memory: " << deviceProp.totalGlobalMem << std::endl;
            std::cout << "Constant memory: " << deviceProp.totalConstMem << std::endl;
            std::cout << "Max mem pitch: " << deviceProp.memPitch << std::endl;
            std::cout << "Texture Alignment: " << deviceProp.textureAlignment << std::endl;
            std::cout << "Multiprocessor count: " << deviceProp.multiProcessorCount << std::endl;
            std::cout << "Shared memory per multiprocessor: " << deviceProp.sharedMemPerBlock << std::endl;
            std::cout << "Registers per mutliprocessor: " << deviceProp.regsPerBlock << std::endl;
            std::cout << "Threads in warp: " << deviceProp.warpSize << std::endl;
            std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "Max thread dimensions: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
            std::cout << "Max grid dimentions: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        }
    }
}