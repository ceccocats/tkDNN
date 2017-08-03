#include <iostream>

#include "Layer.h"

namespace tkDNN {

Conv2d::Conv2d( Network *net, int out_ch, int kernelH, int kernelW, 
                int strideH, int strideW, int paddingH, int paddingW,
                const char* fname_weights, bool batchnorm) : 
    
    LayerWgs(net, net->getOutputDim().c, out_ch, kernelH, kernelW, 1, 
             fname_weights, batchnorm) {

    this->kernelH = kernelH;
    this->kernelW = kernelW;
    this->strideH = strideH;
    this->strideW = strideW;
    this->paddingH = paddingH;
    this->paddingW = paddingW;

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
                net->dataType, net->tensorFormat, out_ch, input_dim.c, 
                kernelH, kernelW) );

    checkCUDNN( cudnnSetConvolution2dDescriptor(convDesc,
                paddingH, paddingW, // padding
                strideH, strideW, // stride
                1,1, // upscale
                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) );

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

    if(!batchnorm) {
        // bias
        alpha = value_type(1);
        beta  = value_type(1);
        checkCUDNN( cudnnAddTensor(net->cudnnHandle,
                    &alpha, biasTensorDesc, bias_d,
                    &beta, dstTensorDesc, dstData) );
    } else {
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardInference(net->cudnnHandle,
            CUDNN_BATCHNORM_SPATIAL, &one, &zero, 
            dstTensorDesc, dstData, dstTensorDesc, 
            dstData, biasTensorDesc, //same tensor descriptor as bias 
            scales_d, bias_d, mean_d, variance_d, 
            CUDNN_BN_MIN_EPSILON);
    }
    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}
