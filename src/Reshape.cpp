#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Reshape::Reshape(Network *net, dataDim_t new_dim, bool final) : Layer(net, final) {

    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(dnnType)) );

    output_dim.n = new_dim.n;
    output_dim.c = new_dim.c;
    output_dim.h = new_dim.h;
    output_dim.w = new_dim.w;
    output_dim.l = new_dim.l;

}

Reshape::~Reshape() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Reshape::infer(dataDim_t &dim, dnnType* srcData) {

    //transpose per channel

    checkCuda( cudaMemcpy(dstData, srcData, dim.n*dim.c*dim.h*dim.w*sizeof(dnnType), cudaMemcpyDeviceToDevice));

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}}