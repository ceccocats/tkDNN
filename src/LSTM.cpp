#include <iostream>

#include "Layer.h"

namespace tk { namespace dnn {

LSTM::LSTM( Network *net, int hiddensize, std::string fname_weights) :
    Layer(net) {

    checkCUDNN( cudnnCreateFilterDescriptor(&paramDesc));
    checkCUDNN( cudnnCreateRNNDescriptor(&rnnDesc) );
    checkCUDNN( cudnnCreateRNNDataDescriptor(&rnnDataDesc) );
    checkCUDNN( cudnnCreateDropoutDescriptor(&dropDesc));

    int n = input_dim.n;
    int c = input_dim.c;
    int h = input_dim.h;
    int w = input_dim.w;
    checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                           net->tensorFormat, net->dataType, n, 1, h, w) );

    int numlayers = 1;
    checkCUDNN( cudnnSetRNNDescriptor(net->cudnnHandle, rnnDesc, hiddensize, numlayers, dropDesc,
            cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL, cudnnRNNMode_t::CUDNN_LSTM,
            cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD, net->dataType) );

    // find dimension of params
    size_t params_size = 0;
    checkCUDNN( cudnnGetRNNParamsSize(net->cudnnHandle, rnnDesc, srcTensorDesc, &params_size, net->dataType) );
    std::cout<<"Params size bytes: "<<params_size<<", floats:  "<<params_size/4<<"\n";


    int dimW[3] = { int(params_size / sizeof(float)), 1, 1};
    checkCUDNN(cudnnCreateFilterDescriptor(&paramDesc));
    checkCUDNN(cudnnSetFilterNdDescriptor(paramDesc, net->dataType, net->tensorFormat, 3, dimW));
    checkCuda( cudaMalloc(&paramsSpace, params_size) );


    int numlinearlayers = 8;

    for(int i=0; i<numlayers*2; i++) {
        std::cout<<"layer: "<<i<<"\n";
        for(int j=0; j<numlinearlayers; j++) {

            // get weights pointer
            cudnnFilterDescriptor_t linLayerMatDesc;
            checkCUDNN(cudnnCreateFilterDescriptor(&linLayerMatDesc));
            dnnType *linLayerMat;

            checkCUDNN(cudnnGetRNNLinLayerMatrixParams(net->cudnnHandle, rnnDesc,
                    i, srcTensorDesc, paramDesc, paramsSpace,
                    j, linLayerMatDesc, (void **)&linLayerMat));

            if(linLayerMat == nullptr) {
                FatalError("LSTM No weights in hidden layer");
            }

            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int nbDims;
            int filterDimA[3];
            checkCUDNN(cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType,
                                                     &format, &nbDims, filterDimA));
            std::cout<<"Wgs Dims: "<<nbDims<<"  ("<<filterDimA[0]<<", "<<filterDimA[1]<<", "<<filterDimA[2]<<")\n";

            // here we should fill the params data into linLayerMat

            checkCUDNN(cudnnDestroyFilterDescriptor(linLayerMatDesc));

            // get bias pointer
            cudnnFilterDescriptor_t linLayerBiasDesc;
            checkCUDNN(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
            float *linLayerBias;

            checkCUDNN(cudnnGetRNNLinLayerBiasParams(net->cudnnHandle, rnnDesc,
                    i, srcTensorDesc, paramDesc, paramsSpace,
                    j, linLayerBiasDesc, (void **)&linLayerBias));

            if(linLayerMat == nullptr) {
                FatalError("LSTM No bias in hidden layer");
            }

            checkCUDNN(cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, &dataType,
                                                     &format, &nbDims, filterDimA));
            std::cout<<"bias Dims: "<<nbDims<<"  ("<<filterDimA[0]<<", "<<filterDimA[1]<<", "<<filterDimA[2]<<")\n";

            // here we should fill the params data into linLayerBiasDesc

            checkCUDNN(cudnnDestroyFilterDescriptor(linLayerBiasDesc));

        }
    }




    checkCUDNN( cudnnCreateTensorDescriptor(&hiddenStateTensorDesc));
    checkCUDNN( cudnnSetTensor4dDescriptor(hiddenStateTensorDesc,
                                           net->tensorFormat, net->dataType, 2*n, c, h, w) );
    checkCuda( cudaMalloc(&hiddenStateData, 2*input_dim.tot()*sizeof(dnnType)) );
    checkCUDNN( cudnnCreateTensorDescriptor(&cellStateTensorDesc));
    checkCUDNN( cudnnSetTensor4dDescriptor(cellStateTensorDesc,
                                           net->tensorFormat, net->dataType, 2*n, c, h, w) );
    checkCuda( cudaMalloc(&cellStateData, 2*input_dim.tot()*sizeof(dnnType)) );


    output_dim = input_dim;
    output_dim.c = hiddensize*2;
    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                           net->tensorFormat, net->dataType, output_dim.n, output_dim.c, output_dim.h, output_dim.w) );




    //allocate data for infer result
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

LSTM::~LSTM() {

    checkCuda( cudaFree(dstData) );
}

dnnType* LSTM::infer(dataDim_t &dim, dnnType* srcData) {

    checkCUDNN(cudnnRNNForwardInference(
        net->cudnnHandle, rnnDesc, 1,
        &srcTensorDesc, srcData,
        hiddenStateTensorDesc, hiddenStateData,
        cellStateTensorDesc, cellStateData,
        paramDesc, paramsSpace,
        &dstTensorDesc, dstData,
        hiddenStateTensorDesc, hiddenStateData,
        cellStateTensorDesc, cellStateData,
        workSpace, ws_sizeInBytes
    ));

    return dstData;
}

}}
