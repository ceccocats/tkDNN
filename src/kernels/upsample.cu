#include "kernels.h"

__global__ void upsample_kernel(size_t N, dnnType *x, int w, int h, int c, int batch, int stride, int forward, float scale, dnnType *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if(forward) out[out_index] += scale * x[in_index];
    else atomicAdd(x+in_index, scale * out[out_index]);
}

void upsampleForward(dnnType* srcData, dnnType* dstData, 
                     int n, int c, int h, int w, int s, int forward, float scale,                                   
                     cudaStream_t stream) {

    int size = w*h*c*n*s*s;
    int blocks = (size+255)/256;
    int threads = 256;
    upsample_kernel<<<blocks, threads, 0, stream>>>(size, srcData, w, h, c, n, s, forward, scale, dstData);
}
