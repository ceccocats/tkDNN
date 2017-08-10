#include "utils.h"

void activationELUForward(dnnType* srcData, dnnType* dstData, int size);
void activationLEAKYForward(dnnType* srcData, dnnType* dstData, int size);
void activationLOGISTICForward(dnnType* srcData, dnnType* dstData, int size);

void reorgForward(  dnnType* srcData, dnnType* dstData, 
                    int n, int c, int h, int w, int stride);
void softmaxForward(float *input, int n, int batch, int batch_offset, 
                    int groups, int group_offset, int stride, float temp, float *output);
