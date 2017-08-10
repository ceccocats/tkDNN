#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"

void activationELUForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream = cudaStream_t(0));
void activationLEAKYForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream = cudaStream_t(0));
void activationLOGISTICForward(dnnType* srcData, dnnType* dstData, int size, cudaStream_t stream = cudaStream_t(0));

void reorgForward(  dnnType* srcData, dnnType* dstData, 
                    int n, int c, int h, int w, int stride, cudaStream_t stream = cudaStream_t(0));
void softmaxForward(float *input, int n, int batch, int batch_offset, 
                    int groups, int group_offset, int stride, float temp, float *output, cudaStream_t stream = cudaStream_t(0));

#endif //KERNELS_H
