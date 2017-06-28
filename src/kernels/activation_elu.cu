#include "kernels.h"

/**
    Exponential Linear Unit compute kernel
    it does the following operation for each x input element:
        x < 0 :    y = e^(x) -1
        x > 0 :    y = x
*/
__global__
void activation_elu(value_type *input, value_type *output, int size) {

    int i = threadIdx.x*(blockIdx.x +1);
    
    if(i<size) 
        output[i] =  (input[i]>0)*input[i] + (input[i]<0)*(expf(input[i]) -1);
    
    // the if x > or < is condensed in one operation for better threads flow 
 }


/**
    ELU activation function
*/
void activationELUForward(value_type* srcData, value_type* dstData, int size)
{
    activation_elu<<<(size+255)/256, 256>>>(srcData, dstData, size);
    checkCuda( cudaDeviceSynchronize() );
}