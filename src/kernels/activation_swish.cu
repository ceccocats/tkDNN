#include "kernels.h"

__global__
void activation_swish(dnnType *input, dnnType *output, int size) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        output[i] =  input[i] * 1.0f/(1.0f + exp(-input[i]));;
    }
 }


/**
    Swish activation function
*/
void activationSwishForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_swish<<<blocks, threads, 0, stream>>>(srcData, dstData, size);
}
