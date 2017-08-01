#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Reorg::Reorg(Network *net, int stride) : Layer(net) {

    this->stride = stride;
    
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c*stride*stride;
    output_dim.h = input_dim.h/stride;
    output_dim.w = input_dim.w/stride;
    output_dim.l = input_dim.l;
    
    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(value_type)) );
}

Reorg::~Reorg() {

    checkCuda( cudaFree(dstData) );
}

value_type* Reorg::infer(dataDim_t &dim, value_type* srcData) {

    reorgForward(srcData, dstData, dim.n, dim.c, dim.h, dim.w, stride);

    dim = output_dim;
    return dstData;
}

}
