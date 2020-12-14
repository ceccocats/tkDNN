#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {
ScaleChannels::ScaleChannels(Network *net, Layer *backLayer, int scale_wh) : Layer(net) {

    this->backLayer = backLayer;
    this->scale_wh = scale_wh;
    output_dim = backLayer->output_dim;
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );

    if( backLayer->output_dim.c != input_dim.c )
        FatalError("ScaleChannels dim missmatch");
    
}

ScaleChannels::~ScaleChannels() {

    checkCuda( cudaFree(dstData) );
}

dnnType* ScaleChannels::infer(dataDim_t &dim, dnnType* srcData) {

    int size = output_dim.n * output_dim.c * output_dim.h * output_dim.w;
    int channel_size = output_dim.h * output_dim.w;
    int batch_size = output_dim.c * output_dim.h * output_dim.w;
    scaleChannelsForward(this->backLayer->dstData, size, channel_size, batch_size, scale_wh, srcData, dstData);

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}}