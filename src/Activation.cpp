#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Activation::Activation(Network *net, dataDim_t input_dim, tkdnnActivationMode_t act_mode) : 
    Layer(net, input_dim) {

    this->act_mode = act_mode;
    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(value_type)) );

    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                        net->tensorFormat,
                                        net->dataType,
                                        input_dim.n, input_dim.c,
                                        input_dim.h, input_dim.w) );
    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                            net->tensorFormat,
                                            net->dataType,
                                            input_dim.n, input_dim.c,
                                            input_dim.h, input_dim.w) );
}

Activation::~Activation() {

    checkCuda( cudaFree(dstData) );
}

value_type* Activation::infer(dataDim_t &dim, value_type* srcData) {

    if(act_mode == ACTIVATION_ELU) {
        activationELUForward(srcData, dstData, dim.tot());

    } else {
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnActivationForward(net->cudnnHandle,
                                            cudnnActivationMode_t(act_mode),
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            dstData) );    
    }
    return dstData;
}

}