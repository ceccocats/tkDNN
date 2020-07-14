#ifndef KERNELSTHRUST_H
#define KERNELSTHRUST_H


#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/copy.h>

#include "tkdnn.h"

struct threshold : public thrust::binary_function<float,float,float>
{
  __host__ __device__
  float operator()(float x, float y) { 
    double toll = 1e-6; 
    if(fabsf(x-y)>toll) 
        return 0.0f; 
    else
        return x; 
    }
};

void sort(dnnType *src_begin, dnnType *src_end, int *idsrc);
void topk(dnnType *src_begin, int *idsrc, int K, float *topk_scores,
            int *topk_inds, float *topk_ys, float *topk_xs);
// void sortAndTopKonDevice(dnnType *src_begin, int *idsrc, float *topk_scores, int *topk_inds, float *topk_ys, float *topk_xs, const int size, const int K, const int n_classes);            
void normalize(float *bgr, const int ch, const int h, const int w, const float *mean, const float *stddev);
void subtractWithThreshold(dnnType *src_begin, dnnType *src_end, dnnType *src2_begin, dnnType *src_out, struct threshold op);
void topKxyclasses(int *ids_begin, int *ids_end, const int K, const int size, const int wh, int *clses, int *xs, int *ys);
void topKxyAddOffset(int * ids_begin, const int K, const int size, int *intxs_begin, int *intys_begin, 
                     float *xs_begin, float *ys_begin, dnnType *src_begin, float *src_out, int *ids_out);
void bboxes(int * ids_begin, const int K, const int size, float *xs_begin, float *ys_begin, 
            dnnType *src_begin, float *bbx0, float *bbx1, float *bby0, float *bby1, float *src_out, int *ids_out);

#endif //KERNELSTHRUST_H