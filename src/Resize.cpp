#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Resize::Resize(Network *net, int scale_c, int scale_h, int scale_w, bool fixed, ResizeMode_t mode) : Layer(net) {

    this->mode = mode;
    if(fixed){
        output_dim.c = scale_c;
        output_dim.h = scale_h;
        output_dim.w = scale_w;
    }
    else{
        output_dim.c *= scale_c;
        output_dim.h *= scale_h;
        output_dim.w *= scale_w;
    }

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

Resize::~Resize() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Resize::infer(dataDim_t &dim, dnnType* srcData) {

    resizeForward(srcData, dstData, dim.n, dim.c, dim.h, dim.w,
                   output_dim.c, output_dim.h, output_dim.w);
    dim = output_dim;

    return dstData;
}

}}