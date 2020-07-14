#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Activation::Activation(Network *net, int act_mode, const float ceiling) : 
    Layer(net) {

    this->act_mode = act_mode;
    this->ceiling = ceiling;
    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(dnnType)) );

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
                                                ceiling) );
    }
}

Activation::~Activation() {

    checkCuda( cudaFree(dstData) );

    if(int(act_mode) < 100)
        checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
}

dnnType* Activation::infer(dataDim_t &dim, dnnType* srcData) {
    if(act_mode == ACTIVATION_LEAKY) {
        activationLEAKYForward(srcData, dstData, dim.tot());

    }
    else if(act_mode == ACTIVATION_MISH) {
        activationMishForward(srcData, dstData, dim.tot());

    } else {
        dnnType alpha = dnnType(1);
        dnnType beta  = dnnType(0);
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

}}
