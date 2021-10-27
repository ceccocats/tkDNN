#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Flatten::Flatten(Network *net) : Layer(net) {

    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(dnnType)) );

    output_dim.n = 1;
    output_dim.c = input_dim.tot();
    output_dim.h = 1;
    output_dim.w = 1;
    output_dim.l = 1;

    this->h = 1;
    this->w = 1;
    this->rows = input_dim.w;
    this->cols = input_dim.h * input_dim.c;
    this->c = input_dim.w * input_dim.h * input_dim.c;
}

Flatten::~Flatten() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Flatten::infer(dataDim_t &dim, dnnType* srcData) {

    //transpose per channel
    matrixTranspose(net->cublasHandle, srcData, dstData, dim.c, dim.h*dim.w*dim.l);

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}}