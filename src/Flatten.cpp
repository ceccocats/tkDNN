#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Flatten::Flatten(Network *net, dataDim_t input_dim) : 
    Layer(net, input_dim) {

    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(value_type)) );

    output_dim.n = 1;
    output_dim.c = input_dim.tot();
    output_dim.h = 1;
    output_dim.w = 1;
    output_dim.l = 1;

}

Flatten::~Flatten() {

    checkCuda( cudaFree(dstData) );
}

value_type* Flatten::infer(dataDim_t &dim, value_type* srcData) {

    //transpose per channel
    matrixTranspose(net->cublasHandle, srcData, dstData, dim.c, dim.h*dim.w*dim.l);

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}