#include "kernelsThrust.h"

__global__
void normalize_kernel(float *bgr, const int dim, const float *mean, const float *stddev){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int j = blockIdx.y;
    bgr[j*(dim)+i] = bgr[j*(dim)+i] - mean[j];
    bgr[j*(dim)+i] = bgr[j*(dim)+i] / stddev[j];
    
}

void normalize(float *bgr, const int ch, const int h, const int w, const float *mean, const float *stddev){
    int num_thread = 256;
    dim3 dimBlock(h*w/num_thread, ch);
    normalize_kernel<<<dimBlock, num_thread, 0>>>(bgr, h*w, mean, stddev);
}