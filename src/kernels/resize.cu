#include "kernels.h"
#include <stdio.h>

__global__ void resize_kernel(  int size,float *x, int i_w, int i_h, int i_c,  
                                int o_w, int o_h, int o_c, int batch, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    int i = id % o_w;
    id /= o_w;
    int j = id % o_h;
    id /= o_h;
    int k = id % o_c;
    id /= o_c;
    int b = id % batch;

    int out_index = i + o_w*(j + o_h*(k + o_c*b));
    int add_index = i/(o_w/i_w) + i_w*(j/(o_h/i_h) + i_h*(k + i_c*b));
    out[out_index] = x[add_index];
}


void resizeForward(  dnnType* srcData, dnnType* dstData, int n, int i_c, int i_h, int i_w,
     int o_c, int o_h, int o_w, cudaStream_t stream )
{
    int o_size = n*o_c*o_h*o_w;

    int blocks = (o_size+255)/256;
    int threads = 256;

    resize_kernel<<<blocks, threads, 0, stream>>>(o_size, srcData, i_w, i_h, i_c, o_w, o_h, o_c, n, dstData);
}
