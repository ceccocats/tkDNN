#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"

void activationELUForward(dnnType *srcData, dnnType *dstData, int size, cudaStream_t stream = cudaStream_t(0));
void activationLEAKYForward(dnnType *srcData, dnnType *dstData, int size, cudaStream_t stream = cudaStream_t(0));
void activationReLUCeilingForward(dnnType *srcData, dnnType *dstData, int size, const float ceiling, cudaStream_t stream = cudaStream_t(0));
void activationLOGISTICForward(dnnType *srcData, dnnType *dstData, int size, cudaStream_t stream = cudaStream_t(0));
void activationSIGMOIDForward(dnnType *srcData, dnnType *dstData, int size, cudaStream_t stream = cudaStream_t(0));
void activationMishForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream= cudaStream_t(0));

void fill(dnnType *data, int size, dnnType val, cudaStream_t stream = cudaStream_t(0));

void resizeForward(dnnType *srcData, dnnType *dstData, int n, int i_c, int i_h, int i_w,
                   int o_c, int o_h, int o_w, cudaStream_t stream = cudaStream_t(0));

void reorgForward(dnnType *srcData, dnnType *dstData,
                  int n, int c, int h, int w, int stride, cudaStream_t stream = cudaStream_t(0));

void MaxPoolingForward(dnnType *srcData, dnnType *dstData, int n, int c, int h, int w, int stride_x, int stride_y, int size, int padding, cudaStream_t stream = cudaStream_t(0));

void softmaxForward(float *input, int n, int batch, int batch_offset,
                    int groups, int group_offset, int stride, float temp, float *output, cudaStream_t stream = cudaStream_t(0));

void shortcutForward(dnnType *srcData, dnnType *dstData, int n1, int c1, int h1, int w1, int s1,
                     int n2, int c2, int h2, int w2, int s2,
                     cudaStream_t stream = cudaStream_t(0));

void upsampleForward(dnnType *srcData, dnnType *dstData,
                     int n, int c, int h, int w, int s, int forward, float scale,
                     cudaStream_t stream = cudaStream_t(0));

void float2half(float *srcData, __half *dstData, int size, const cudaStream_t stream = cudaStream_t(0));

void dcnV2CudaForward(cublasStatus_t stat, cublasHandle_t handle,
                         float *input, float *weight,
                         float *bias, float *ones,
                         float *offset, float *mask,
                         float *output, float *columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group, const int batch_id,
                         const int in_n, const int in_c, const int in_h, const int in_w,
                         const int out_n, const int out_c, const int out_h, const int out_w,
                         const int dst_dim, cudaStream_t stream = cudaStream_t(0));

void scalAdd(dnnType* dstData, int size, float alpha, float beta, int inc, cudaStream_t stream = cudaStream_t(0));
#endif //KERNELS_H
