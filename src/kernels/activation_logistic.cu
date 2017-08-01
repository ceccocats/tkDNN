#include "kernels.h"

__global__
void activation_logistic(value_type *input, value_type *output, int size) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        output[i] =  1.0f/(1.0f + exp(-input[i]));;
    }
 }


/**
    LOGISTIC activation function
*/
void activationLOGISTICForward(value_type* srcData, value_type* dstData, int size)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_logistic<<<blocks, threads>>>(srcData, dstData, size);
    checkCuda( cudaDeviceSynchronize() );
}


