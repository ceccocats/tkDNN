#include <iostream>

#include "Layer.h"

namespace tkDNN {

Layer::Layer(Network *net) {

    this->net = net;
    this->input_dim = net->getOutputDim();
    this->output_dim = input_dim;
    
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