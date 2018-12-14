#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Softmax::Softmax(Network *net) : Layer(net) {

    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(dnnType)) );

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

dnnType* Softmax::infer(dataDim_t &dim, dnnType* srcData) {

    dnnType alpha = dnnType(1);
    dnnType beta  = dnnType(0);
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

}}
