#include <iostream>

#include "Layer.h"

namespace tkDNN {

LayerWgs::LayerWgs(Network *net, dataDim_t in_dim, 
                int inputs, int outputs, int kh, int kw, int kl, 
                const char* fname_weights, bool batchnorm) : Layer(net, in_dim) {

    this->inputs  = inputs;
    this->outputs = outputs;    
    this->weights_path  = std::string(fname_weights);
   
    std::cout<<"Reading weights: I="<<inputs<<" O="<<outputs<<" KERNEL="<<kh<<"x"<<kw<<"x"<<kl<<"\n";
    int seek = 0;
    readBinaryFile(weights_path.c_str(), inputs*outputs*kh*kw*kl, &data_h, &data_d, seek);
    seek += inputs*outputs*kh*kw*kl;
    readBinaryFile(weights_path.c_str(), outputs, &bias_h, &bias_d, seek);
    
    this->batchnorm = batchnorm;
    if(batchnorm) {
        seek += outputs;
        readBinaryFile(weights_path.c_str(), outputs, &scales_h, &scales_d, seek);
        seek += outputs;
        readBinaryFile(weights_path.c_str(), outputs, &mean_h, &mean_d, seek);
        seek += outputs;
        readBinaryFile(weights_path.c_str(), outputs, &variance_h, &variance_d, seek);
    }
}

LayerWgs::~LayerWgs() {

    delete [] data_h;
    delete [] bias_h;
    checkCuda( cudaFree(data_d) );
    checkCuda( cudaFree(bias_d) );

    if(batchnorm) {
        delete [] scales_h;
        delete [] mean_h;
        delete [] variance_h;
        checkCuda( cudaFree(scales_d) );
        checkCuda( cudaFree(mean_d) );
        checkCuda( cudaFree(variance_d) );
    }
}

}
