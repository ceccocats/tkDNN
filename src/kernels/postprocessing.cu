#include "kernelsThrust.h"

void transformDep(float *src_begin, float *src_end, float *dst_begin, float *dst_end) {
    int e = exp(-6);
    thrust::transform(thrust::device, dst_begin, dst_end, thrust::make_constant_iterator(e), dst_begin, thrust::plus<float>());
    thrust::transform(thrust::device, src_begin, src_end, dst_begin, dst_begin, thrust::divides<float>());
    thrust::transform(thrust::device, dst_begin, dst_end, thrust::make_constant_iterator(-1.0), dst_begin, thrust::plus<float>());
}

void subtractWithThreshold(dnnType *src_begin, dnnType *src_end, dnnType *src2_begin, dnnType *src_out, struct threshold op){
    thrust::transform(thrust::device, src_begin, src_end, src2_begin, src_out, op);
}

void sort(dnnType *src_begin, dnnType *src_end, int *idsrc){
    thrust::sort_by_key(thrust::device,
        src_begin, src_end, idsrc,
        thrust::greater<float>());
    // thrust::stable_sort_by_key(thrust::device,
    //     src_begin, src_end, idsrc,
    //     thrust::greater<float>());
}

void topk(dnnType *src_begin, int *idsrc, int K, float *topk_scores,
            int *topk_inds, float *topk_ys, float *topk_xs){
    checkCuda( cudaMemcpy(topk_scores, (float *)src_begin, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    checkCuda( cudaMemcpy(topk_inds, idsrc, K*sizeof(int), cudaMemcpyDeviceToDevice) );    
}

__global__ 
void sortAndTopK_kernel(dnnType *src_begin, int *idsrc, float *topk_scores, int *topk_inds, float *topk_ys, float *topk_xs,const int size, const int K){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    thrust::sort_by_key(thrust::device, src_begin + i * size, src_begin + i * size + size, idsrc + i * size, thrust::greater<float>());
    thrust::copy_n(thrust::device, src_begin + i * size, K, topk_scores + i * K);
    thrust::copy_n(thrust::device, idsrc + i * size, K, topk_inds + i * K );
}

void sortAndTopKonDevice(dnnType *src_begin, int *idsrc, float *topk_scores, int *topk_inds, float *topk_ys, float *topk_xs, const int size, const int K, const int n_classes){
    int blocks = n_classes;
    int threads = 1;
    sortAndTopK_kernel<<<blocks, threads, 0>>>(src_begin, idsrc, topk_scores, topk_inds, topk_ys, topk_xs, size, K);   
}

__global__ 
void maxElem_kernel(float *src_begin, float *dst_begin, const int n_classes, const int size){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i > size)
        return;

    thrust::device_ptr<float> dPbeg ( &src_begin[i*n_classes] ) ;
    thrust::device_ptr<float> dPend = dPbeg + n_classes;
    thrust::device_ptr<float> result = thrust::max_element(thrust::device,dPbeg, dPend);

    dst_begin[i] = result - dPbeg;
}

void maxElem(dnnType *src_begin, dnnType *dst_begin, const int c, const int h, const int w){
    int blocks = (h*w)/32+1;
    int threads = 32;
    maxElem_kernel<<<blocks, threads, 0>>>(src_begin, dst_begin, c, h*w);   
}

void topKxyclasses(int *ids_begin, int *ids_end, const int K, const int size, const int wh, int *clses, int *xs, int *ys){    
    thrust::transform(thrust::device, ids_begin, ids_end, thrust::make_constant_iterator(wh), clses, thrust::divides<int>());
    thrust::transform(thrust::device, ids_begin, ids_end, thrust::make_constant_iterator(wh), ids_begin, thrust::modulus<int>());
    thrust::transform(thrust::device, ids_begin, ids_end, thrust::make_constant_iterator(size), ys, thrust::divides<int>());
    thrust::transform(thrust::device, ids_begin, ids_end, thrust::make_constant_iterator(size), xs, thrust::modulus<int>());
}

void topKxyAddOffset(int * ids_begin, const int K, const int size, 
                     int *intxs_begin, int *intys_begin, float *xs_begin, 
                     float *ys_begin, dnnType *src_begin, float *src_out, int *ids_out){
    thrust::gather(thrust::device, ids_begin, ids_begin + K, src_begin, src_out);
    thrust::transform(thrust::device, intxs_begin, intxs_begin + K, src_out, xs_begin, thrust::plus<float>());
    thrust::transform(thrust::device, ids_begin, ids_begin + K, thrust::make_constant_iterator(size), ids_out, thrust::plus<int>());
    thrust::gather(thrust::device, ids_out, ids_out+K, src_begin, src_out);
    thrust::transform(thrust::device, intys_begin, intys_begin + K, src_out, ys_begin, thrust::plus<float>());
}

void getRecordsFromTopKId(int * ids_begin, const int K, const int ch, const int size, dnnType *src_begin, float *src_out, int *ids_out) {
    for(int i=0; i<ch; i++) {
        // thrust::gather(thrust::device, ids_begin, ids_begin + K, src_begin, src_out);
        thrust::transform(thrust::device, ids_begin, ids_begin + K, thrust::make_constant_iterator(i*size), ids_out, thrust::plus<int>());
        thrust::gather(thrust::device, ids_out, ids_out + K, src_begin, src_out+i*K);
    }
}

void bboxes(int * ids_begin, const int K, const int size, float *xs_begin, float *ys_begin, 
            dnnType *src_begin, float *bbx0, float *bbx1, float *bby0, float *bby1,
            float *src_out, int *ids_out){ 
    thrust::gather(thrust::device, ids_begin, ids_begin + K, src_begin, src_out);
    thrust::transform(thrust::device, src_out, src_out + K, thrust::make_constant_iterator(2), src_out, thrust::divides<float>());
    // x0
    thrust::transform(thrust::device, xs_begin, xs_begin + K, src_out, bbx0, thrust::minus<float>());
    // x1
    thrust::transform(thrust::device, xs_begin, xs_begin + K, src_out, bbx1, thrust::plus<float>());
    thrust::transform(thrust::device, ids_begin, ids_begin + K, thrust::make_constant_iterator(size), ids_out, thrust::plus<int>());
    thrust::gather(thrust::device, ids_out, ids_out + K, src_begin, src_out);
    thrust::transform(thrust::device, src_out, src_out + K, thrust::make_constant_iterator(2), src_out, thrust::divides<float>());
    // y0
    thrust::transform(thrust::device, ys_begin, ys_begin + K, src_out, bby0, thrust::minus<float>());
    // y1
    thrust::transform(thrust::device, ys_begin, ys_begin + K, src_out, bby1, thrust::plus<float>());
}

