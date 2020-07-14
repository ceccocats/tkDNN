#include "kernels.h"
#include <stdio.h>
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

__global__ void resize_kernel(  int i_N,float *x, int i_w, int i_h, int i_c,  
                                int o_w, int o_h, int o_c, int batch, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= i_N) return;

    int out_index = i;
    int out_w = i%o_w;
    i = i/o_w;
    int out_h = i%o_h;
    i = i/o_h;
    int out_c = i%o_c;
    i = i/o_c;

    //copying last column/last row as padding
    int in_index = ((i*i_c + MIN(out_c,i_c-1))*i_h +  MIN(out_h,i_h-1))*i_w + MIN(out_w, i_w-1);
    out[out_index] = x[in_index];
}


void resizeForward(  dnnType* srcData, dnnType* dstData, int n, int i_c, int i_h, int i_w,
     int o_c, int o_h, int o_w, cudaStream_t stream )
{
    int i_size = n*i_c*i_h*i_w;
    int o_size = n*o_c*o_h*o_w;

    int blocks = (o_size+255)/256;
    int threads = 256;

    if(i_c == o_c && i_h == o_h && i_w == o_w )
    {
        checkCuda(cudaMemcpy(dstData, srcData, i_size*sizeof(dnnType), cudaMemcpyDeviceToDevice));
    }
    else
    {
        checkCuda(cudaMemset(dstData, 0, o_size*sizeof(dnnType)));
        resize_kernel<<<blocks, threads, 0, stream>>>(o_size, srcData, i_w, i_h, i_c, o_w, o_h, o_c, n, dstData);
        // printDeviceVector(i_size, srcData);
        // printDeviceVector(o_size, dstData);
    }
}
