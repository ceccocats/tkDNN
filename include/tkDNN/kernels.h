#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"

TKDNN_LIB_EXPORT_API void activationELUForward(dnnType *srcData, dnnType *dstData, int size, cudaStream_t stream = cudaStream_t(0));
TKDNN_LIB_EXPORT_API void activationLEAKYForward(dnnType *srcData, dnnType *dstData, int size, float slope, cudaStream_t stream = cudaStream_t(0));
TKDNN_LIB_EXPORT_API void activationReLUCeilingForward(dnnType *srcData, dnnType *dstData, int size, const float ceiling, cudaStream_t stream = cudaStream_t(0));
TKDNN_LIB_EXPORT_API void activationLOGISTICForward(dnnType *srcData, dnnType *dstData, int size, cudaStream_t stream = cudaStream_t(0));
TKDNN_LIB_EXPORT_API void activationSIGMOIDForward(dnnType *srcData, dnnType *dstData, int size, cudaStream_t stream = cudaStream_t(0));
TKDNN_LIB_EXPORT_API void activationMishForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream= cudaStream_t(0));

TKDNN_LIB_EXPORT_API void fill(dnnType *data, int size, dnnType val, cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void resizeForward(dnnType *srcData, dnnType *dstData, int n, int i_c, int i_h, int i_w,
                   int o_c, int o_h, int o_w, cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void reorgForward(dnnType *srcData, dnnType *dstData,
                  int n, int c, int h, int w, int stride, cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void MaxPoolingForward(dnnType *srcData, dnnType *dstData, int n, int c, int h, int w, int stride_x, int stride_y, int size, int padding, cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void softmaxForward(float *input, int n, int batch, int batch_offset,
                    int groups, int group_offset, int stride, float temp, float *output, cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void shortcutForward(dnnType *srcData, dnnType *dstData, int n1, int c1, int h1, int w1, int s1,
                     int n2, int c2, int h2, int w2, int s2, bool mul,
                     cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void upsampleForward(dnnType *srcData, dnnType *dstData,
                     int n, int c, int h, int w, int s, int forward, float scale,
                     cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void float2half(float *srcData, __half *dstData, int size, const cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void dcnV2CudaForward(cublasStatus_t stat, cublasHandle_t handle,
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

TKDNN_LIB_EXPORT_API void scalAdd(dnnType* dstData, int size, float alpha, float beta, int inc, cudaStream_t stream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void reflection_pad2d_out_forward(int32_t pad_h,int32_t pad_w,float *srcData,float *dstData,int32_t input_h,int32_t input_w,int32_t plane_dim,int32_t n_batch,cudaStream_t cudaStream = cudaStream_t(0));

TKDNN_LIB_EXPORT_API void constant_pad2d_forward(dnnType *srcData,dnnType *dstData,int32_t input_h,int32_t input_w,int32_t output_h,
                            int32_t output_w,int32_t c,int32_t n,int32_t padT,int32_t padL,dnnType constant,cudaStream_t cudaStream = cudaStream_t(0));


#endif //KERNELS_H
