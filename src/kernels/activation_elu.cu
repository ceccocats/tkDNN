#include "kernels.h"

__global__
void activation_elu(value_type *input, value_type *output, int size) {

    int i = threadIdx.x*(blockIdx.x +1);
    
    if(i<size) 
        output[i] =  (input[i]>0)*input[i] + (input[i]<0)*(expf(input[i]) -1);
 }


void activationELUForward(value_type* srcData, value_type* dstData, int size)
{
    activation_elu<<<(size+255)/256, 256>>>(srcData, dstData, size);
    checkCuda( cudaDeviceSynchronize() );
}