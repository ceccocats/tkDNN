#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Pooling::Pooling( Network *net, int winH, int winW, int strideH, int strideW,
                  int paddingH, int paddingW,
                  tkdnnPoolingMode_t pool_mode) : 
    Layer(net) {

    this->winH = winH;
    this->winW = winW;
    this->strideH = strideH;
    this->strideW = strideW;
    this->pool_mode = pool_mode;
    this->paddingH = paddingH;
    this->paddingW = paddingW;

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

    cudnnPoolingMode_t cudnn_pool_mode = cudnnPoolingMode_t(pool_mode);
    if(pool_mode == POOLING_MAX_FIXEDSIZE) cudnn_pool_mode = cudnnPoolingMode_t(tkdnnPoolingMode_t::POOLING_MAX);

    checkCUDNN( cudnnSetPooling2dDescriptor(poolingDesc, cudnn_pool_mode,
                CUDNN_NOT_PROPAGATE_NAN, winH, winW, paddingH, paddingW, strideH, strideW) );

    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc, 
                net->tensorFormat, net->dataType, n, c, h, w) );

    //get out dim
    // checkCUDNN( cudnnGetPooling2dForwardOutputDim(poolingDesc, srcTensorDesc, &n, &c, &h, &w)); 

    //compute w and h as in darknet
    if(pool_mode == tkdnnPoolingMode_t::POOLING_MAX_FIXEDSIZE){
        int padH = paddingH == 0? winH -1 : paddingH;
        int padW = paddingW == 0? winW -1 : paddingW;
        h = (h + padH - winH)/strideH +1;
        w =  (w + padW - winW)/strideW +1;
    }
    else{
        h = (h + 2*paddingH - winH)/strideH +1 ;
        w =  (w + 2*paddingW - winW)/strideW +1;
    }
    
    // h = (h + winH*this->paddingH)/strideH;
    // w = (w + winW*this->paddingW)/strideW;

    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                net->tensorFormat, net->dataType, n, c, h, w) );
       
    output_dim.n = n;
    output_dim.c = c;
    output_dim.h = h;
    output_dim.w = w;
    output_dim.l = l;

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );

    //pool on 3d data need transposition at the enter and on the exit
    //allocate for initial and final transposition
    if(poolOn3d) {
        output_dim.n = 1;

        checkCuda( cudaMalloc(&tmpInputData, input_dim.tot()*sizeof(dnnType)) );
        checkCuda( cudaMalloc(&tmpOutputData, output_dim.tot()*sizeof(dnnType)) );
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

dnnType* Pooling::infer(dataDim_t &dim, dnnType* srcData) {

    dnnType *poolSrc = srcData;
    dnnType *poolDst = dstData;

    if(poolOn3d) {
        matrixTranspose(net->cublasHandle, srcData, tmpInputData, dim.h*dim.w*dim.c, dim.l);
        poolSrc = tmpInputData;
        poolDst = tmpOutputData;
    }

    if(pool_mode == tkdnnPoolingMode_t::POOLING_MAX_FIXEDSIZE){
        MaxPoolingForward(poolSrc, poolDst, dim.n, dim.c, dim.h, dim.w, this->strideH, this->strideW, this->winH, this->winH-1);
    }
    else{
        dnnType alpha = dnnType(1);
        dnnType beta = dnnType(0);
        checkCUDNN( cudnnPoolingForward(net->cudnnHandle, poolingDesc,
                    &alpha, srcTensorDesc, poolSrc,
                    &beta, dstTensorDesc, poolDst) );
    }

    //update dim
    dim = output_dim;

    if(poolOn3d)
        matrixTranspose(net->cublasHandle, tmpOutputData, dstData, dim.l, dim.h*dim.w*dim.c);

    return dstData;
}

}}
