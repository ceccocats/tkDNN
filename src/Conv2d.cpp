#include <iostream>

#include "Layer.h"

namespace tkDNN {

Conv2d::Conv2d( Network *net, dataDim_t in_dim, int out_ch, 
                int kernelH, int kernelW, int strideH, int strideW,
                const char* fname_weights, const char* fname_bias) : 
    
    LayerWgs(net, in_dim, in_dim.c, out_ch, kernelH, kernelW, 1, 
             fname_weights, fname_bias) {

    this->kernelH = kernelH;
    this->kernelW = kernelW;
    this->strideH = strideH;
    this->strideW = strideW;

    checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
    checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );

    int n = input_dim.n;
    int c = input_dim.c;
    int h = input_dim.h;
    int w = input_dim.w;

    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                net->tensorFormat, net->dataType, n, c, h, w) );

    checkCUDNN( cudnnSetFilter4dDescriptor(filterDesc,
                net->dataType, out_ch, input_dim.c, 
                kernelH, kernelW) );

    checkCUDNN( cudnnSetConvolution2dDescriptor(convDesc,
                0,0, // padding
                strideH, strideW, // stride
                1,1, // upscale
                CUDNN_CROSS_CORRELATION) );

    // find dimension of convolution output
    checkCUDNN( cudnnGetConvolution2dForwardOutputDim(
                convDesc, srcTensorDesc, filterDesc,
                &n, &c, &h, &w) );

    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                net->tensorFormat, net->dataType, n, c, h, w) );
                
    checkCUDNN( cudnnGetConvolutionForwardAlgorithm(net->cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo) );
 
    workSpace = NULL;
    ws_sizeInBytes = 0;

    checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(net->cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                algo, &ws_sizeInBytes) );

    if (ws_sizeInBytes!=0) {
        checkCuda( cudaMalloc(&workSpace, ws_sizeInBytes) );
    }


    checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                net->tensorFormat, net->dataType,
                1, out_ch, 1, 1) );


    output_dim.n = n;
    output_dim.c = c;
    output_dim.h = h;
    output_dim.w = w;
    output_dim.l = 1;

    //allocate data for infer result
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(value_type)) );
}

Conv2d::~Conv2d() {
    
    checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
    checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );

    if (ws_sizeInBytes!=0)
        checkCuda( cudaFree(workSpace) );

    checkCuda( cudaFree(dstData) );
}

value_type* Conv2d::infer(dataDim_t &dim, value_type* srcData) {


    // convolution
    value_type alpha = value_type(1);
    value_type beta  = value_type(0);
    checkCUDNN( cudnnConvolutionForward(net->cudnnHandle,
                &alpha, srcTensorDesc, srcData, filterDesc,
                data_d, convDesc, algo, workSpace, ws_sizeInBytes,
                &beta, dstTensorDesc, dstData) );

    // bias
    alpha = value_type(1);
    beta  = value_type(1);
    checkCUDNN( cudnnAddTensor(net->cudnnHandle, CUDNN_ADD_SAME_C,
                &alpha, biasTensorDesc, bias_d,
                &beta, dstTensorDesc, dstData) );

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}