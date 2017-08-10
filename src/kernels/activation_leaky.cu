#include "kernels.h"

__global__
void activation_leaky(dnnType *input, dnnType *output, int size) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        if (input[i]>0)
            output[i] = input[i];
        else
            output[i] = 0.1f*input[i];
    }
 }


/**
    ELU activation function
*/
void activationLEAKYForward(dnnType* srcData, dnnType* dstData, int size)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_leaky<<<blocks, threads>>>(srcData, dstData, size);
    checkCuda( cudaDeviceSynchronize() );
}


