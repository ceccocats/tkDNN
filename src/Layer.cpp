#include <iostream>

#include "Layer.h"

namespace tkDNN {

Layer::Layer(Network *net, dataDim_t in_dim) {

    this->net = net;
    this->input_dim = in_dim;
    this->output_dim = in_dim;
    
    checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );

    if(!net->addLayer(this))
        FatalError("Net reached max number of layers");    
}

Layer::~Layer() {

    checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
}

}