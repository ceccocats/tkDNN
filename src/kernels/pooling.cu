#include "kernels.h"

__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *input, float *output)
{
    int h = (in_h + pad - size) / stride_y + 1;
    int w = (in_w + pad - size) / stride_x + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad / 2;
    int h_offset = -pad / 2;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -9999999;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride_y + l;
            int cur_w = w_offset + j*stride_x + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -9999999;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
}

void MaxPoolingForward(dnnType* srcData, dnnType* dstData, int n, int c, int h, int w, int stride_x, int stride_y, int size, int padding, cudaStream_t stream) 
{

    int tot_size = n*c*h*w;

    int blocks = (tot_size+255)/256;
    int threads = 256;
    
    forward_maxpool_layer_kernel<<<blocks, threads, 0, stream>>>(tot_size, h, w, c, stride_x, stride_y, size, padding, srcData, dstData);
}

