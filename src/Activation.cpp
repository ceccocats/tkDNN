#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Activation::Activation(Network *net, int act_mode) : 
    Layer(net) {

    this->act_mode = act_mode;
    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(value_type)) );

    if(int(act_mode) < 100) {

        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                            net->tensorFormat,
                                            net->dataType,
                                            input_dim.n*input_dim.l, 
                                            input_dim.c,
                                            input_dim.h, input_dim.w) );
        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                net->tensorFormat,
                                                net->dataType,
                                                input_dim.n*input_dim.l, 
                                                input_dim.c,
                                                input_dim.h, input_dim.w) );


        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                (cudnnActivationMode_t) act_mode,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );
    }
}

Activation::~Activation() {

    checkCuda( cudaFree(dstData) );

    if(int(act_mode) < 100)
        checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
}

value_type* Activation::infer(dataDim_t &dim, value_type* srcData) {

    if(act_mode == ACTIVATION_LEAKY) {
        activationLEAKYForward(srcData, dstData, dim.tot());
    
    } else {
        value_type alpha = value_type(1);
        value_type beta  = value_type(0);
        checkCUDNN( cudnnActivationForward(net->cudnnHandle,
                                            activDesc,
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
