#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"

void activationELUForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream = cudaStream_t(0));
void activationLEAKYForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream = cudaStream_t(0));
void activationLOGISTICForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream = cudaStream_t(0));

void fill(dnnType* data, int size, dnnType val, cudaStream_t stream = cudaStream_t(0));

void reorgForward(  dnnType* srcData, dnnType* dstData, 
                    int n, int c, int h, int w, int stride, cudaStream_t stream = cudaStream_t(0));
void softmaxForward(float *input, int n, int batch, int batch_offset, 
                    int groups, int group_offset, int stride, float temp, float *output, cudaStream_t stream = cudaStream_t(0));


void shortcutForward(dnnType* srcData, dnnType* dstData, int n1, int c1, int h1, int w1, int s1,
                                                         int n2, int c2, int h2, int w2, int s2, 
                     cudaStream_t stream = cudaStream_t(0));

void upsampleForward(dnnType* srcData, dnnType* dstData, 
                     int n, int c, int h, int w, int s, int forward, float scale,                                   
                     cudaStream_t stream = cudaStream_t(0));

void float2half(float* srcData, __half* dstData, int size, const cudaStream_t stream = cudaStream_t(0));


void modulated_deformable_im2col_cuda(cudaStream_t stream,
                                const float *data_im, const float *data_offset, const float *data_mask,
                                const int batch_size, const int channels, const int height_im, const int width_im,
                                const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
                                const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                const int dilation_h, const int dilation_w,
                                const int deformable_group, float *data_col);
#endif //KERNELS_H
