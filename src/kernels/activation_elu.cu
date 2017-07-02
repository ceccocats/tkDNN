#include "kernels.h"

/**
    Exponential Linear Unit compute kernel
    it does the following operation for each x input element:
        x < 0 :    y = e^(x) -1
        x > 0 :    y = x
*/
__global__
void activation_elu(value_type *input, value_type *output, int size) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        value_type k0, k1;
        
        if (input[i]>0)
            k0 = 1.0f;
        else
            k0 = 0.0f;
        k1 = 1.0f-k0;

        output[i] = k0*input[i] + k1*(expf(input[i]) -1.0f);
    }
 }


/**
    ELU activation function
*/
void activationELUForward(value_type* srcData, value_type* dstData, int size)
{
    int blocks = (size+255)/256;
    int threads = 256;
    
    activation_elu<<<blocks, threads>>>(srcData, dstData, size);
    checkCuda( cudaDeviceSynchronize() );
}