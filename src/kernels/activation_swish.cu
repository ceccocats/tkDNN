#include "kernels.h"
#include <math.h>

// https://github.com/AlexeyAB/darknet/blob/master/src/activation_kernels.cu
__device__ float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}

__global__
void activation_swish(dnnType *input, dnnType *output, int size) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < size) 
    {
        float x_val = input[i];
        float sigmoid = logistic_activate_kernel(x_val);
        output[i] = x_val * sigmoid;
    }
}

/**
    swish activation function
*/
void activationSwishForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_swish<<<blocks, threads, 0, stream>>>(srcData, dstData, size);
}