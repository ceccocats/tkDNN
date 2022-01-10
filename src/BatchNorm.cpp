#include <iostream>

#include "Layer.h"

namespace tk { namespace dnn { 
    void BatchNorm::initCUDNN(){
        cudnnTensorDescriptor_t srcTensor = srcTensorDesc;
        cudnnTensorDescriptor_t dstTensor = dstTensorDesc;
        dataDim_t idim,odim;
        idim = input_dim;
        odim = output_dim;

        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensor,
                                           net->tensorFormat, net->dataType, idim.n, idim.c, idim.h, idim.w) );

        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        
        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensor,
                                           net->tensorFormat, net->dataType, odim.n, odim.c, odim.h, odim.w) );

        checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                                           net->tensorFormat, net->dataType,
                                           1, output_dim.c, 1, 1) );


    }

    void BatchNorm::inferCUDNN(float *srcData){
        dnnType alpha = dnnType(1);
        dnnType beta  = dnnType(0);

        alpha = dnnType(1);
        beta  = dnnType(1);
        checkCUDNN( cudnnAddTensor(net->cudnnHandle,
                                   &alpha, biasTensorDesc, bias_d,
                                   &beta, dstTensorDesc, dstData) );
        alpha = dnnType(1);
        beta  = dnnType(0);
        checkCUDNN( cudnnBatchNormalizationForwardInference(net->cudnnHandle,
                                                    CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                                                    dstTensorDesc, dstData, dstTensorDesc,
                                                    dstData, biasTensorDesc, //same tensor descriptor as bias
                                                    scales_d, bias_d, mean_d, variance_d,
                                                    TKDNN_BN_MIN_EPSILON) );
    }

    BatchNorm::BatchNorm(Network *net,int output,std::string fname_weights) : 
        LayerBNWgs(net,net->getOutputDim().c,output,fname_weights){
            output_dim = input_dim;
            initCUDNN();
            checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );

    }

    dnnType* BatchNorm::infer(dataDim_t &dim,dnnType* srcData){
        inferCUDNN(srcData);

        dim = output_dim;
        return dstData;
    }

    BatchNorm::~BatchNorm(){
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCuda( cudaFree(dstData) );
    }

}}