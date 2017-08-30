#include "kernels.h"

__global__
void float2half_device(float *input, __half *output, int size) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        output[i] = __float2half(input[i]);
    }
}


void float2half(float* srcData, __half *dstData, int size, const cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    float2half_device<<<blocks, threads, 0, stream>>>(srcData, dstData, size);
    cudaDeviceSynchronize();
}
