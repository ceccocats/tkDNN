#include "utils.h"

void activationELUForward(value_type* srcData, value_type* dstData, int size);
void activationLEAKYForward(value_type* srcData, value_type* dstData, int size);
void activationLOGISTICForward(value_type* srcData, value_type* dstData, int size);

void reorgForward(  value_type* srcData, value_type* dstData, 
                    int n, int c, int h, int w, int stride);
void softmaxForward(float *input, int n, int batch, int batch_offset, 
                    int groups, int group_offset, int stride, float temp, float *output);
