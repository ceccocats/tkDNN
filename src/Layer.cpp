#include <iostream>

#include "Layer.h"

namespace tkDNN {

Layer::Layer(Network *net, dataDim_t in_dim) {

    this->net = net;
    this->input_dim = in_dim;
    this->output_dim = in_dim;
    
    checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
}

Layer::~Layer() {

    checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
}

}