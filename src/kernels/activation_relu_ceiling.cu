#include "kernels.h"

__global__
void activation_relu_ceiling(dnnType *input, dnnType *output, int size, const float ceiling) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        if (input[i]>0)
        {
            if (input[i]>ceiling)
                output[i] = ceiling;
            else
                output[i] = input[i];
        }
        else
            output[i] = 0.0f;
    }
 }


/**
    Relu ceiling activation function
*/
void activationReLUCeilingForward(dnnType* srcData, dnnType* dstData, int size, const float ceiling, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_relu_ceiling<<<blocks, threads, 0, stream>>>(srcData, dstData, size, ceiling);
}


