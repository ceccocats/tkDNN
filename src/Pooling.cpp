#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Pooling::Pooling(   Network *net, dataDim_t input_dim, 
                    int winH, int winW, int strideH, int strideW, tkdnnPoolingMode_t pool_mode) : 
    Layer(net, input_dim) {


    if(winH != strideH || winW != strideW)
        FatalError("stride pooling not yet implemented");

    this->winH = winH;
    this->winW = winW;
    this->strideH = strideH;
    this->strideW = strideW;
    this->pool_mode = pool_mode;
    
    checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );

    int n = input_dim.n;
    int c = input_dim.c;
    int h = input_dim.h;
    int w = input_dim.w;
    int l = input_dim.l;

    poolOn3d = false;
    
    if(l > 1) { 
        poolOn3d = true;
    
        if(n != 1)
            FatalError("N value on 3d pool must be 1");

        //use batch as l
        n = l;
    }

    checkCUDNN( cudnnSetPooling2dDescriptor(poolingDesc, cudnnPoolingMode_t(pool_mode),
                CUDNN_NOT_PROPAGATE_NAN, winH, winW, 0, 0, strideH, strideW) );

    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc, 
                net->tensorFormat, net->dataType, n, c, h, w) );

    //get out dim
    h = h / winH; w = w / winW;
    
    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                net->tensorFormat, net->dataType, n, c, h, w) );
       

    output_dim.n = n;
    output_dim.c = c;
    output_dim.h = h;
    output_dim.w = w;
    output_dim.l = l;

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(value_type)) );

    //pool on 3d data need transposition at the enter and on the exit
    //allocate for initial and final transposition
    if(poolOn3d) {
        output_dim.n = 1;

        checkCuda( cudaMalloc(&tmpInputData, input_dim.tot()*sizeof(value_type)) );
        checkCuda( cudaMalloc(&tmpOutputData, output_dim.tot()*sizeof(value_type)) );
    }

}

Pooling::~Pooling() {

    if(poolOn3d) {
        checkCuda( cudaFree(tmpInputData) );
        checkCuda( cudaFree(tmpOutputData) );
    }

    checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
    checkCuda( cudaFree(dstData) );
}

value_type* Pooling::infer(dataDim_t &dim, value_type* srcData) {

    value_type *poolSrc = srcData;
    value_type *poolDst = dstData;

    if(poolOn3d) {
        matrixTranspose(net->cublasHandle, srcData, tmpInputData, dim.h*dim.w*dim.c, dim.l);
        poolSrc = tmpInputData;
        poolDst = tmpOutputData;
    }

    value_type alpha = value_type(1);
    value_type beta = value_type(0);
    checkCUDNN( cudnnPoolingForward(net->cudnnHandle, poolingDesc,
                &alpha, srcTensorDesc, poolSrc,
                &beta, dstTensorDesc, poolDst) );

    //update dim
    dim = output_dim;

    if(poolOn3d)
        matrixTranspose(net->cublasHandle, tmpOutputData, dstData, dim.l, dim.h*dim.w*dim.c);

    return dstData;
}

}
