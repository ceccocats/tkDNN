#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Shortcut::Shortcut(Network *net, Layer *backLayer) : Layer(net) {

    this->backLayer = backLayer;
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );

    if( /*backLayer->output_dim.c != input_dim.c ||*/
        backLayer->output_dim.w != input_dim.w ||
        backLayer->output_dim.h != input_dim.h   )
        FatalError("Shortcut dim mismatch");
}

Shortcut::~Shortcut() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Shortcut::infer(dataDim_t &dim, dnnType* srcData) {

    dataDim_t bdim = this->backLayer->output_dim;

    checkCuda(cudaMemcpy(dstData, srcData, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));
    shortcutForward(this->backLayer->dstData, dstData, dim.n, dim.c, dim.h, dim.w, 1, bdim.n, bdim.c, bdim.h, bdim.w, 1);

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}}