#include "kernels.h"

__device__ 
__forceinline__ 
double sigmoid (double a)
{
    return 1.0 / (1.0 + exp (-a));
}


__global__
void activation_sigmoid(dnnType *input, dnnType *output, int size) {

    int stride = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid; i < size; i += stride) {
        output[i] = sigmoid (input[i]);
    }
 }


/**
    ELU activation function
*/
void activationSIGMOIDForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_sigmoid<<<blocks, threads, 0, stream>>>(srcData, dstData, size);
}