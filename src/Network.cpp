#include <iostream>

#include "tkdnn.h"
#include "Network.h"
#include "Layer.h"

namespace tkDNN {

Network::Network(dataDim_t input_dim) {
    this->input_dim = input_dim;

    float tk_ver = float(tkDNN::getVersion())/1000;
    float cu_ver = float(cudnnGetVersion())/1000;

    std::cout<<"New NETWORK (tkDNN v"<<tk_ver
             <<", CUDNN v"<<cu_ver<<")\n";
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
    for(int i=0; i<num_layers; i++) {
        data = layers[i]->infer(dim, data);
        //dim.print();
    }
    checkCuda(cudaDeviceSynchronize());
    return data;
}

bool Network::addLayer(Layer *l) {
    if(num_layers == MAX_LAYERS)
        return false;
    
    layers[num_layers++] = l;
    return true;
}

dataDim_t Network::getOutputDim() {

        if(num_layers == 0)
            return input_dim;
        else
            return layers[num_layers-1]->output_dim;
}

}
