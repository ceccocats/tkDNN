#include <iostream>

#include "Layer.h"

namespace tkDNN {

Conv3d::Conv3d( Network *net, dataDim_t in_dim, int out_ch,
                int kernelH, int kernelW, int kernelL, 
                int strideH, int strideW, int strideL,
                const char* fname_weights, const char* fname_bias) : 
    
    LayerWgs(net, in_dim, in_dim.c, out_ch, kernelH, kernelW, kernelL, 
             fname_weights, fname_bias) {

    this->kernelH = kernelH;
    this->kernelW = kernelW;
    this->kernelL = kernelL;
    this->strideH = strideH;
    this->strideW = strideW;
    this->strideL = strideL;

    checkCUDNN( cudnnCreateTensorDescriptor(&biasDstTensorDesc) );
    checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
    checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );

    int n = input_dim.n;
    int c = input_dim.c;
    int h = input_dim.h;
    int w = input_dim.w;
    int l = input_dim.l;

    //create a tensor Nd descriptor with N = 4  
    int dimA[5];
    dimA[0] = n; dimA[1] = c; dimA[2] = h; dimA[3] = w; dimA[4] = l;
    int strideA[5];
    strideA[0] = c*h*w*l; 
    strideA[1] = h*w*l; 
    strideA[2] = w*l; 
    strideA[3] = l; 
    strideA[4] = 1;
    checkCUDNN( cudnnSetTensorNdDescriptor(srcTensorDesc, 
                net->dataType, 5, dimA, strideA));

    //filter descriptor
    int filterDim[5];
    filterDim[0] = out_ch;     
    filterDim[1] = in_dim.c; 
    filterDim[2] = kernelH; 
    filterDim[3] = kernelW;
    filterDim[4] = kernelL;
    checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc, 
                net->dataType, 5, filterDim));

    //convolutional descriptor
    int padA[3] = {0, 0, 0};
    int filterStride[3] = {strideH, strideW, strideL};
    int upscale[3] = {1, 1, 1};
    checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc, 3, 
                padA, filterStride, upscale, CUDNN_CROSS_CORRELATION));


    //get output dimension
    int outputDim[5];
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, 5, outputDim));
    n = outputDim[0];
    c = outputDim[1];
    h = outputDim[2];
    w = outputDim[3];
    l = outputDim[4];

    //destination sensor
    int outputStride[5];
    outputStride[0] = c*h*w*l;
    outputStride[1] = h*w*l;
    outputStride[2] = w*l;
    outputStride[3] = l;
    outputStride[4] = 1;
    checkCUDNN( cudnnSetTensorNdDescriptor(dstTensorDesc, net->dataType,
                5, outputDim, outputStride));


    //conv algo
    checkCUDNN( cudnnGetConvolutionForwardAlgorithm(net->cudnnHandle,
                srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo) );
   
    checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(net->cudnnHandle,
                                            srcTensorDesc,
                                            filterDesc,
                                            convDesc,
                                            dstTensorDesc,
                                            algo,
                                            &ws_sizeInBytes) );
    if (ws_sizeInBytes!=0) 
        checkCuda( cudaMalloc(&workSpace, ws_sizeInBytes) );
    

    // bias on N dimensional is not SUPPORTED so i have to use 2d method
    //the trick is to upscale the 2d matrix width by the factor of 3d thickness 
    checkCUDNN( cudnnSetTensor4dDescriptor(biasDstTensorDesc,
                net->tensorFormat, net->dataType, n, c, h*l, w) );


    checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                net->tensorFormat, net->dataType,
                1, c, 1, 1) );

    output_dim.n = n;
    output_dim.c = c;
    output_dim.h = h;
    output_dim.w = w;
    output_dim.l = l;

    //allocate data for infer result
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(value_type)) );
}

Conv3d::~Conv3d() {
    
    checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
    checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(biasDstTensorDesc) );

    if (ws_sizeInBytes!=0)
        checkCuda( cudaFree(workSpace) );

    checkCuda( cudaFree(dstData) );
}

value_type* Conv3d::infer(dataDim_t &dim, value_type* srcData) {


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
                &beta, biasDstTensorDesc, dstData) );

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}