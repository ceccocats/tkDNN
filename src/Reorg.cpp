#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Reorg::Reorg(Network *net, int stride, bool reorg3d) : Layer(net) {

    this->stride = stride;
    this->reorg3d = reorg3d;
    
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c*stride*stride;
    output_dim.h = input_dim.h/stride;
    output_dim.w = input_dim.w/stride;
    output_dim.l = input_dim.l;
    
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

Reorg::~Reorg() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Reorg::infer(dataDim_t &dim, dnnType* srcData) {
    if (reorg3d)
        reorgForward(srcData, dstData, output_dim.n, output_dim.c, output_dim.h, output_dim.w, stride);
    else
        reorgForward(srcData, dstData, dim.n, dim.c, dim.h, dim.w, stride);

    dim = output_dim;
    return dstData;
}

}}
