#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Softmax::Softmax(Network *net, dataDim_t input_dim) : 
    Layer(net, input_dim) {

    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(value_type)) );

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
}

Softmax::~Softmax() {

    checkCuda( cudaFree(dstData) );
}

value_type* Softmax::infer(dataDim_t &dim, value_type* srcData) {

    value_type alpha = value_type(1);
    value_type beta  = value_type(0);
    checkCUDNN( cudnnSoftmaxForward(net->cudnnHandle,
                                    CUDNN_SOFTMAX_ACCURATE ,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha,
                                    srcTensorDesc,
                                    srcData,
                                    &beta,
                                    dstTensorDesc,
                                    dstData) );
    return dstData;
}

}
