#include "kernels.h"
#include <math.h>


__global__
void activation_sigmoid(dnnType *input, dnnType *output, int size) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < size)
        output[i] = 1.0f / (1.0f + exp (-input[i]));
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