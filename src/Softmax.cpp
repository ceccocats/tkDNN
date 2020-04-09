#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Softmax::Softmax(Network *net, const tk::dnn::dataDim_t* dim, const cudnnSoftmaxMode_t mode) : Layer(net) {

    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(dnnType)) );

    this->mode = mode;
    if(dim == nullptr)
    {
        this->dim.n= input_dim.n;
        this->dim.c= input_dim.c;
        this->dim.h= input_dim.h;
        this->dim.w= input_dim.w;
        this->dim.l= input_dim.l;
    }
    else
    {
        this->dim.n= dim->n;
        this->dim.c= dim->c;
        this->dim.h= dim->h;
        this->dim.w= dim->w;
        this->dim.l= dim->l;
    }

    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                        net->tensorFormat,
                                        net->dataType,
                                        this->dim.n*this->dim.l, 
                                        this->dim.c,
                                        this->dim.h, this->dim.w) );
    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                            net->tensorFormat,
                                            net->dataType,
                                            this->dim.n*this->dim.l, 
                                            this->dim.c,
                                            this->dim.h, this->dim.w) );
}

Softmax::~Softmax() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Softmax::infer(dataDim_t &dim, dnnType* srcData) {

    dnnType alpha = dnnType(1);
    dnnType beta  = dnnType(0);
    checkCUDNN( cudnnSoftmaxForward(net->cudnnHandle,
                                    CUDNN_SOFTMAX_ACCURATE ,
                                    this->mode,
                                    &alpha,
                                    srcTensorDesc,
                                    srcData,
                                    &beta,
                                    dstTensorDesc,
                                    dstData) );
    return dstData;
}

}}
