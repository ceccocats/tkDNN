#include <iostream>
#include <string.h>

#include "tkdnn.h"
#include "Network.h"
#include "Layer.h"

namespace tk { namespace dnn {

Network::Network(dataDim_t input_dim) {
    this->input_dim = input_dim;

    float tk_ver = float(TKDNN_VERSION)/1000;
    float cu_ver = float(cudnnGetVersion())/1000;

    std::cout<<"New NETWORK (tkDNN v"<<tk_ver
             <<", CUDNN v"<<cu_ver<<")\n";
    dataType = CUDNN_DATA_FLOAT;
    tensorFormat = CUDNN_TENSOR_NCHW;

    checkCUDNN( cudnnCreate(&cudnnHandle) );
    checkERROR( cublasCreate(&cublasHandle) );

    num_layers = 0;

    fp16 = false;
    if(const char* env_p = std::getenv("TKDNN_MODE"))
        if(strcmp(env_p, "FP16") == 0)
            fp16 = true;
   
    if(fp16)
        std::cout<<COL_REDB<<"!! FP16 INERENCE ENABLED !!"<<COL_END<<"\n";
}

Network::~Network() {

    checkCUDNN( cudnnDestroy(cudnnHandle) );
    checkERROR( cublasDestroy(cublasHandle) );
}

dnnType* Network::infer(dataDim_t &dim, dnnType* data) {

    //do infer for every layer
    for(int i=0; i<num_layers; i++) {
        data = layers[i]->infer(dim, data);
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

void Network::print() {

    printCenteredTitle(" NETWORK MODEL ", '=', 60);
    std::cout.width(3); std::cout<<std::left<<"N.";
    std::cout<<" ";
    std::cout.width(17); std::cout<<std::left<<"Layer type";
    std::cout.width(22); std::cout<<std::left<<"input (H*W,CH)";
    std::cout.width(16); std::cout<<std::left<<"output (H*W,CH)";
    std::cout<<"\n";

    for(int i=0; i<num_layers; i++) {
        dataDim_t in = layers[i]->input_dim;
        dataDim_t out = layers[i]->output_dim;

        std::cout.width(3); std::cout<<std::right<<i;
        std::cout<<" ";
        std::cout.width(16); std::cout<<std::left<<layers[i]->getLayerName();
        std::cout.width(4);  std::cout<<std::right<<in.h;
        std::cout<<" x ";
        std::cout.width(4);  std::cout<<std::right<<in.w;
        std::cout<<", ";
        std::cout.width(4);  std::cout<<std::right<<in.c;
        std::cout<<"  -> ";
        std::cout.width(4);  std::cout<<std::right<<out.h;
        std::cout<<" x ";
        std::cout.width(4);  std::cout<<std::right<<out.w;
        std::cout<<", ";
        std::cout.width(4);  std::cout<<std::right<<out.c;
        std::cout<<"\n";
    }
    printCenteredTitle("", '=', 60);
    std::cout<<"\n";
}


}}
