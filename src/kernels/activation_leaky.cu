#include "kernels.h"

__global__
void activation_leaky(dnnType *input, dnnType *output, int size, float slope) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        if (input[i]>0)
            output[i] = input[i];
        else
            output[i] = slope*input[i];
    }
 }


/**
    ELU activation function
*/
void activationLEAKYForward(dnnType* srcData, dnnType* dstData, int size, float slope, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_leaky<<<blocks, threads, 0, stream>>>(srcData, dstData, size, slope);
}


