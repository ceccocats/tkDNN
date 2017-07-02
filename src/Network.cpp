#include <iostream>

#include "Network.h"
#include "Layer.h"

namespace tkDNN {

Network::Network() {

    std::cout<<"New NETWORK with CUDNN v"<<float(cudnnGetVersion())/1000<<"\n";
    dataType = CUDNN_DATA_FLOAT;
    tensorFormat = CUDNN_TENSOR_NCHW;

    checkCUDNN( cudnnCreate(&cudnnHandle) );
    checkERROR( cublasCreate(&cublasHandle) );

    num_layers = 0;
}

Network::~Network() {

    checkCUDNN( cudnnDestroy(cudnnHandle) );
    checkERROR( cublasDestroy(cublasHandle) );
}

value_type* Network::infer(dataDim_t &dim, value_type* data) {

    //do infer for every layer
    for(int i=0; i<num_layers; i++)
        data = layers[i]->infer(dim, data);

    return data;
}

bool Network::addLayer(Layer *l) {
    if(num_layers == MAX_LAYERS)
        return false;
    
    layers[num_layers++] = l;
    return true;
}

}