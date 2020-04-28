#include "kernels.h"
#include <math.h>

__global__ void scal_add_kernel(dnnType* dstData, int size, float alpha, float beta, int inc)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < size) dstData[i*inc] = dstData[i*inc] * alpha + beta;
}

void scalAdd(dnnType* dstData, int size, float alpha, float beta, int inc, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    scal_add_kernel<<<blocks, threads, 0, stream>>>(dstData, size, alpha, beta, inc);
}