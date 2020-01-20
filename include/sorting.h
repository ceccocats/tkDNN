#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/copy.h>


#include "tkdnn.h"

void sort(dnnType *src_begin, dnnType *src_end, int *idsrc);
void topk(dnnType *src_begin, int *idsrc, int K, float *topk_scores,
            int *topk_inds, float *topk_ys, float *topk_xs);
void sortAndTopKonDevice(dnnType *src_begin, int *idsrc, float *topk_scores, int *topk_inds, float *topk_ys, float *topk_xs, const int size, const int K, const int n_classes);            
void subtractWithThreshold(dnnType *src_begin, dnnType *src_end, dnnType *src2_begin, dnnType *src_out);
void topKxyclasses(int *ids_begin, int *ids_end, const int K, const int size, const int wh, int *clses, int *xs, int *ys);
void topKxyAddOffset(int * ids_begin, const int K, const int size, int *intxs_begin, int *intys_begin, float *xs_begin, float *ys_begin, dnnType *src_begin);
void bboxes(int * ids_begin, const int K, const int size, float *xs_begin, float *ys_begin, dnnType *src_begin, float *bbx0, float *bbx1, float *bby0, float *bby1);
