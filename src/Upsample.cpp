#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Upsample::Upsample(Network *net, int stride) : Layer(net) {

    this->stride = stride;
    
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c*stride*stride;
    output_dim.h = input_dim.h/stride;
    output_dim.w = input_dim.w/stride;
    output_dim.l = input_dim.l;
    
    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(dnnType)) );
}

Upsample::~Upsample() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Upsample::infer(dataDim_t &dim, dnnType* srcData) {


    dim = output_dim;
    return dstData;
}

}}
