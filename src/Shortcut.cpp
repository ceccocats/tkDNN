#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Shortcut::Shortcut(Network *net, Layer *backLayer, int layers_n) : Layer(net) {

    this->backLayer = backLayer;
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

Shortcut::~Shortcut() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Shortcut::infer(dataDim_t &dim, dnnType* srcData) {




    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}}