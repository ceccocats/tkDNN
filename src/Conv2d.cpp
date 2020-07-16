#include <iostream>

#include "Layer.h"

namespace tk { namespace dnn {

void Conv2d::initCUDNN(bool back) {

    cudnnTensorDescriptor_t srcTensor = srcTensorDesc;
    cudnnTensorDescriptor_t dstTensor = dstTensorDesc;

    dataDim_t idim, odim;
    if(!back) {
        idim = input_dim;
        odim = output_dim;
    } else {
        idim = output_dim;
        odim = input_dim;
    }

    checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
    checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
    checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );

    // input tensor dim
    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensor,
                                           net->tensorFormat, net->dataType, idim.n, idim.c, idim.h, idim.w) );

    checkCUDNN( cudnnSetFilter4dDescriptor(filterDesc,
                                           net->dataType, net->tensorFormat, odim.c, idim.c/groups,
                                           kernelH, kernelW) );

    checkCUDNN( cudnnSetConvolution2dDescriptor(convDesc,
                                                paddingH, paddingW, // padding
                                                strideH, strideW, // stride
                                                1,1, // upscale
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) );

    checkCUDNN(  cudnnSetConvolutionGroupCount(convDesc,
                                                groups) );

    // check dimension of convolution output
    dataDim_t tmpdim;
    checkCUDNN( cudnnGetConvolution2dForwardOutputDim(
            convDesc, srcTensor, filterDesc,
            &tmpdim.n, &tmpdim.c, &tmpdim.h, &tmpdim.w) );

    if(odim.n != tmpdim.n || odim.c != tmpdim.c || odim.h != tmpdim.h || odim.w != tmpdim.w) {
        std::cout<<"tkdim input: "; idim.print();
        std::cout<<"tkdim output: "; odim.print();
        std::cout<<"cudnndim: "; tmpdim.print();
        FatalError("Error conv dimension mismatch");
    }

    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensor,
                                           net->tensorFormat, net->dataType, odim.n, odim.c, odim.h, odim.w) );

    checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                                           net->tensorFormat, net->dataType,
                                           1, output_dim.c, 1, 1) );

    // init workspace
    workSpace = NULL;
    ws_sizeInBytes = 0;
    int algo_count = 0;
    if(back) {
        checkCUDNN( cudnnGetConvolutionBackwardDataAlgorithm_v7(net->cudnnHandle,
                                                                filterDesc, dstTensor, convDesc, srcTensor, 1, &algo_count, &bwAlgo) );
        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(net->cudnnHandle,
                                                                filterDesc, dstTensor, convDesc, srcTensor,
                                                                bwAlgo.algo, &ws_sizeInBytes));
        

        // invert tensors
        srcTensorDesc = dstTensor;
        dstTensorDesc = srcTensor;
    } else {
        
        checkCUDNN( cudnnGetConvolutionForwardAlgorithm_v7(net->cudnnHandle,
                                                         srcTensor, filterDesc, convDesc, dstTensor,
                                                        1, &algo_count, &algo) );
         checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(net->cudnnHandle,
                                                            srcTensor, filterDesc, convDesc, dstTensor,
                                                           algo.algo, &ws_sizeInBytes));
    }

    if(algo_count < 1)
        FatalError("Cannot retrieve convolutional algo");
}

void Conv2d::inferCUDNN(dnnType* srcData, bool back) {

    dnnType alpha = dnnType(1);
    dnnType beta  = dnnType(0);
    if(back) {
        checkCUDNN(cudnnConvolutionBackwardData(net->cudnnHandle,
                                                &alpha, filterDesc, data_d,
                                                srcTensorDesc, srcData,
                                                convDesc, bwAlgo.algo, workSpace, ws_sizeInBytes,
                                                &beta, dstTensorDesc, dstData));
    } else {
        checkCUDNN(cudnnConvolutionForward(net->cudnnHandle,
                                           &alpha, srcTensorDesc, srcData, filterDesc,
                                           data_d, convDesc, algo.algo, workSpace, ws_sizeInBytes,
                                           &beta, dstTensorDesc, dstData));
    }

    if(!batchnorm && !additional_bias) { //CHECK WITH IF CORRECT
        // bias
        alpha = dnnType(1);
        beta  = dnnType(1);
        checkCUDNN( cudnnAddTensor(net->cudnnHandle,
                                   &alpha, biasTensorDesc, bias_d,
                                   &beta, dstTensorDesc, dstData) );
    } else {
        if(additional_bias)
        {
            alpha = dnnType(1);
            beta  = dnnType(1);
            checkCUDNN( cudnnAddTensor(net->cudnnHandle,
                                        &alpha, biasTensorDesc, bias2_d,
                                        &beta, dstTensorDesc, dstData) );
        }
        if(batchnorm)
        {
            alpha = dnnType(1);
            beta  = dnnType(0);
            checkCUDNN( cudnnBatchNormalizationForwardInference(net->cudnnHandle,
                                                    CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                                                    dstTensorDesc, dstData, dstTensorDesc,
                                                    dstData, biasTensorDesc, //same tensor descriptor as bias
                                                    scales_d, bias_d, mean_d, variance_d,
                                                    TKDNN_BN_MIN_EPSILON) );
        }
    }
}

Conv2d::Conv2d( Network *net, int out_ch, int kernelH, int kernelW,
                int strideH, int strideW, int paddingH, int paddingW,
                std::string fname_weights, bool batchnorm, bool deConv, int groups, bool additional_bias) :
    
    LayerWgs(net, net->getOutputDim().c, out_ch, kernelH, kernelW, 1, 
             fname_weights, batchnorm, additional_bias, deConv, groups) {
    this->kernelH = kernelH;
    this->kernelW = kernelW;
    this->strideH = strideH;
    this->strideW = strideW;
    this->paddingH = paddingH;
    this->paddingW = paddingW;
    this->deConv = deConv;
    this->groups = groups;
    this->additional_bias = additional_bias;

    if(!deConv) {
        output_dim.n = input_dim.n;
        output_dim.c = out_ch;
        output_dim.h = (input_dim.h + 2 * paddingH - kernelH) / strideH + 1;
        output_dim.w = (input_dim.w + 2 * paddingW - kernelW) / strideW + 1;
        output_dim.l = 1;
    } else {
        output_dim.n = input_dim.n;
        output_dim.c = out_ch;
        output_dim.h = ((input_dim.h-1) * strideH) - 2*paddingH + kernelH;
        output_dim.w = ((input_dim.w-1) * strideW) - 2*paddingW + kernelW;
        output_dim.l = 1;
    }
    initCUDNN(deConv);

    // allocate warkspace
    if (ws_sizeInBytes!=0) {
        checkCuda( cudaMalloc(&workSpace, ws_sizeInBytes) );
    }

    //allocate data for infer result
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

Conv2d::~Conv2d() {

    checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
    checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
    checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );

    if (ws_sizeInBytes!=0)
        checkCuda( cudaFree(workSpace) );

    checkCuda( cudaFree(dstData) );
}

dnnType* Conv2d::infer(dataDim_t &dim, dnnType* srcData) {

    if(deConv) {
        FatalError("you must use DeConv class for Deconvolutional layers");
    }

    // convolution
    inferCUDNN(srcData, false);

    //update data dimensions    
    dim = output_dim;
    return dstData;
}

dnnType* DeConv2d::infer(dataDim_t &dim, dnnType* srcData) {

    // convolution
    inferCUDNN(srcData, true);

    //update data dimensions
    dim = output_dim;
    return dstData;
}

}}
