#include "kernels.h"
#include <math.h>

#define MISH_THRESHOLD 20

__device__ 
float tanh_activate_kernel(float x){return (2/(1 + expf(-2*x)) - 1);}

__device__ 
float softplus_kernel(float x, float threshold = 20) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return logf(expf(x) + 1);
}



__device__ 
float mish_yashas(float x) { 
    float e = __expf(x); 
    if (x <= -18.0f) 
        return x * e; 
 
    float n = e * e + 2 * e; 
    if (x <= -5.0f) 
        return x * __fdividef(n, n + 2); 
 
    return x - 2 * __fdividef(x, n + 2); 
} 

// https://github.com/digantamisra98/Mish
// https://github.com/AlexeyAB/darknet/blob/master/src/activation_kernels.cu
__global__
void activation_mish(dnnType *input, dnnType *output, int size) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < size) 
        // output[i] = input[i] * tanh_activate_kernel( softplus_kernel(input[i], MISH_THRESHOLD));    
        output[i] = mish_yashas(input[i]);
}

/**
    Mish activation function
*/
void activationMishForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_mish<<<blocks, threads, 0, stream>>>(srcData, dstData, size);
}