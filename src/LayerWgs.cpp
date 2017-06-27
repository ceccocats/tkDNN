#include <iostream>

#include "Layer.h"

namespace tkDNN {

LayerWgs::LayerWgs(Network *net, dataDim_t in_dim, 
                int inputs, int outputs, int kh, int kw, int kl, 
                const char* fname_weights, const char* fname_bias) : Layer(net, in_dim) {

    this->inputs  = inputs;
    this->outputs = outputs;    
    this->weights_path  = std::string(fname_weights);
    this->bias_path     = std::string(fname_bias);

    std::cout<<"Reading weights: I="<<inputs<<" O="<<outputs<<" KERNEL="<<kh<<"x"<<kw<<"x"<<kl<<"\n";
    readBinaryFile(weights_path.c_str(), inputs*outputs*kh*kw*kl, &data_h, &data_d);
    readBinaryFile(bias_path.c_str(), outputs, &bias_h, &bias_d);
}

LayerWgs::~LayerWgs() {

    delete [] data_h;
    delete [] bias_h;
    checkCuda( cudaFree(data_d) );
    checkCuda( cudaFree(bias_d) );
}

}