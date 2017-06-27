#include <iostream>

#include "Network.h"

namespace tkDNN {

Network::Network() {

    std::cout<<"New NETWORK with CUDNN v"<<float(cudnnGetVersion())/1000<<"\n";
    dataType = CUDNN_DATA_FLOAT;
    tensorFormat = CUDNN_TENSOR_NCHW;

    checkCUDNN( cudnnCreate(&cudnnHandle) );
    checkERROR( cublasCreate(&cublasHandle) );
}

Network::~Network() {

    checkCUDNN( cudnnDestroy(cudnnHandle) );
    checkERROR( cublasDestroy(cublasHandle) );
}

}