#include "kernels.h"

__global__
void fill_kernel(dnnType *data, int size, dnnType val) {

    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i<size) {    
        data[i] = val;
    }
 }


void fill(dnnType* data, int size, dnnType val, cudaStream_t stream)
{
    int blocks = (size+255)/256;
    int threads = 256;
    fill_kernel<<<blocks, threads, 0, stream>>>(data, size, val);
}


