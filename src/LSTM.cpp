#include <iostream>

#include "Layer.h"

namespace tk { namespace dnn {

LSTM::LSTM( Network *net, int hiddensize, bool returnSeq, std::string fname_weights) :
    Layer(net) {

    this->returnSeq = returnSeq;
    int batchSize = input_dim.n;
    int inputSize = input_dim.c;
    seqLen        = input_dim.w;
    stateSize     = hiddensize;

    std::cout<<"LSTM seqLen: "<<seqLen<<"\n";

    // init Tensor Descriptors
    std::vector<cudnnTensorDescriptor_t> x_vec(seqLen);
    std::vector<cudnnTensorDescriptor_t> y_vec(seqLen);

    int dimA[3];
    int strideA[3];
    for (int i = 0; i < seqLen; i++) {
        checkCUDNN(cudnnCreateTensorDescriptor(&x_vec[i]));
        checkCUDNN(cudnnCreateTensorDescriptor(&y_vec[i]));

        dimA[0] = batchSize;
        dimA[1] = inputSize;
        dimA[2] = 1;
        dimA[0] = batchSize;
        dimA[1] = inputSize;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;
        checkCUDNN(cudnnSetTensorNdDescriptor(x_vec[i],
            net->dataType, 3, dimA, strideA));

        dimA[0] = batchSize;
        dimA[1] = bidirectional ? stateSize*2 : stateSize;
        dimA[2] = 1;
        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;
        checkCUDNN(cudnnSetTensorNdDescriptor(y_vec[i],
            net->dataType, 3, dimA, strideA));
    }
    // apply tensordesc
    x_desc_vec_ = x_vec;
    y_desc_vec_ = y_vec;


    // set the state tensors
    dimA[0] = numLayers * (bidirectional ? 2 : 1);
    dimA[1] = batchSize;
    dimA[2] = stateSize;
    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;
    checkCUDNN(cudnnCreateTensorDescriptor(&hx_desc_));
    checkCUDNN(cudnnCreateTensorDescriptor(&cx_desc_));
    checkCUDNN(cudnnCreateTensorDescriptor(&hy_desc_));
    checkCUDNN(cudnnCreateTensorDescriptor(&cy_desc_));
    checkCUDNN(cudnnSetTensorNdDescriptor(hx_desc_, net->dataType, 3, dimA, strideA));
    checkCUDNN(cudnnSetTensorNdDescriptor(cx_desc_, net->dataType, 3, dimA, strideA));
    checkCUDNN(cudnnSetTensorNdDescriptor(hy_desc_, net->dataType, 3, dimA, strideA));
    checkCUDNN(cudnnSetTensorNdDescriptor(cy_desc_, net->dataType, 3, dimA, strideA));
    // allocate     dnnType *hx_ptr, *cx_ptr, *hy_ptr, *cy_ptr;
    stateDataDim = dimA[0]*dimA[1]*dimA[2];
    checkCuda( cudaMalloc(&hx_ptr, stateDataDim*sizeof(dnnType)) );
    checkCuda( cudaMalloc(&cx_ptr, stateDataDim*sizeof(dnnType)) );
    checkCuda( cudaMalloc(&hy_ptr, stateDataDim*sizeof(dnnType)) );
    checkCuda( cudaMalloc(&cy_ptr, stateDataDim*sizeof(dnnType)) );
    


    // Create Dropout descriptors // TODO: ??? IS IT NECESSARY ???
    float dropoutprob = 0.1f; // random val ????
    checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
    checkCUDNN(cudnnDropoutGetStatesSize(net->cudnnHandle, &dropout_byte_));
    dropout_size_ = dropout_byte_ / sizeof(dnnType);
    checkCuda( cudaMalloc(&dropout_states_, dropout_byte_) );
    uint64_t seed_ = 17 + rand() % 4096;  // NOLINT(runtime/threadsafe_fn)
    checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc,
        net->cudnnHandle, dropoutprob, dropout_states_, dropout_byte_, seed_));


    // RNN descriptors
    checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc));

    checkCUDNN(cudnnSetRNNDescriptor(net->cudnnHandle, 
        rnnDesc, stateSize, numLayers, dropoutDesc,
        cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
        (bidirectional ? cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL : cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL),
        cudnnRNNMode_t::CUDNN_LSTM,
        cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
        net->dataType));


    // Get temp space sizes
    checkCUDNN(cudnnGetRNNWorkspaceSize(net->cudnnHandle,
        rnnDesc, seqLen, x_desc_vec_.data(), &workspace_byte_));
    workspace_size_ = workspace_byte_ / sizeof(dnnType);
    checkCuda( cudaMalloc(&work_space_, workspace_byte_) );    
    

    // Check that number of params are correct
    size_t cudnn_param_size;
    checkCUDNN(cudnnGetRNNParamsSize(net->cudnnHandle,
        rnnDesc,x_desc_vec_[0], &cudnn_param_size, net->dataType));
    int cudnn_params = cudnn_param_size/sizeof(dnnType);
    std::cout<<"LSTM params size: "<<cudnn_params << ", bytes: "<<cudnn_param_size<<"\n";

    // Set param descriptors
    checkCUDNN(cudnnCreateFilterDescriptor(&w_desc_));
    int dim_w[3] = {1, 1, 1};
    dim_w[0] = cudnn_params;
    checkCUDNN(cudnnSetFilterNdDescriptor(w_desc_,
        net->dataType, net->tensorFormat, 3, dim_w));

    // load params
    readBinaryFile(fname_weights, cudnn_params, &w_h, &w_ptr);
    
    //allocate data for infer result
    int dstDim = input_dim.n * stateSize*(bidirectional ? 2 : 1) * input_dim.h * input_dim.w;
    checkCuda( cudaMalloc(&dstData, dstDim*sizeof(dnnType)) );

    // set output dim 
    output_dim = input_dim;
    output_dim.c = stateSize*(bidirectional ? 2 : 1);
    if(!returnSeq) {
        output_dim.h = 1;
        output_dim.w = 1;
    }




    // Query weight layout
    cudnnFilterDescriptor_t m_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&m_desc));
    dnnType *p;
    int n = 8; // lstm layers
    
    printCenteredTitle("WEIGHTS", '=', 20);
    for (int i = 0; i < numLayers*(bidirectional?2:1); ++i) {
        for (int j = 0; j < n; ++j) {
            
            checkCUDNN(cudnnGetRNNLinLayerMatrixParams(net->cudnnHandle, rnnDesc,
                i, x_desc_vec_[0], w_desc_, 0, j, m_desc, (void**)&p));
            
            std::cout << "ptr: " << ((int64_t)(p - NULL))/sizeof(dnnType)<<"\n";
            
            cudnnDataType_t t;
            cudnnTensorFormat_t f;
            int ndim = 5;
            int dims[5] = {0, 0, 0, 0, 0};
            checkCUDNN(cudnnGetFilterNdDescriptor(m_desc, ndim, &t, &f, &ndim, &dims[0]));
            std::cout << "(layer, linlayer): " <<  i << " " << j << "\n";

            int tot = 1;
            for (int i = 0; i < ndim; ++i) {
                std::cout << dims[i] << " ";
                tot *= dims[i];
            }
            std::cout<<"\t-> "<<tot<<"\n\n";
        }
    }

    printCenteredTitle("BIAS", '=', 20);
    for (int i = 0; i < numLayers*(bidirectional?2:1); ++i) {
        for (int j = 0; j < n; ++j) {
            checkCUDNN(cudnnGetRNNLinLayerBiasParams(net->cudnnHandle, rnnDesc, 
                i, x_desc_vec_[0], w_desc_, 0, j, m_desc, (void**)&p));

            std::cout << "ptr: " << ((int64_t)(p - NULL))/sizeof(dnnType)<<"\n";

            cudnnDataType_t t;
            cudnnTensorFormat_t f;
            int ndim = 5;
            int dims[5] = {0, 0, 0, 0, 0};
            checkCUDNN(cudnnGetFilterNdDescriptor(m_desc, ndim, &t, &f, &ndim, &dims[0]));
            std::cout << "(layer, linlayer): " <<  i << " " << j << "\n";

            int tot = 1;
            for (int i = 0; i < ndim; ++i) {
                std::cout << dims[i] << " ";
                tot *= dims[i];
            }
            std::cout<<"\t-> "<<tot<<"\n\n";
        }
    }
    
    checkCUDNN(cudnnDestroyFilterDescriptor(m_desc));
}

LSTM::~LSTM() {
    checkCuda(cudaFree(hx_ptr));
    checkCuda(cudaFree(cx_ptr));
    checkCuda(cudaFree(hy_ptr));
    checkCuda(cudaFree(cy_ptr));
    checkCuda(cudaFree(w_ptr ));

    checkCuda(cudaFree(work_space_    ));
    checkCuda(cudaFree(dropout_states_));

    checkCuda(cudaFree(dstData));
}

dnnType* LSTM::infer(dataDim_t &dim, dnnType* srcData) {
    std::cout<<"LSTM infer\n";

    // reset states
    checkCuda( cudaMemset(hx_ptr, 0, stateDataDim*sizeof(float)) );	
    checkCuda( cudaMemset(cx_ptr, 0, stateDataDim*sizeof(float)) );


    checkCUDNN(cudnnRNNForwardInference(net->cudnnHandle,
        rnnDesc,
        seqLen,                     // number of time steps (nT)
        x_desc_vec_.data(),         // input array of desc (nT*nC_in)
        srcData,                    // input pointer
        hx_desc_,                   // initial hidden state desc     
        hx_ptr,                     // initial hidden state pointer 
        cx_desc_,                   // initial cell state desc      
        cx_ptr,                     // initial cell state pointer   
        w_desc_,                    // weights desc
        w_ptr,                      // weights pointer
        y_desc_vec_.data(),         // output desc     (nT*nC_out)
        dstData,                    // output pointer
        hy_desc_,                   // final hidden state desc        
        hy_ptr,                     // final hidden state pointer 
        cy_desc_,                   // final cell state desc          
        cy_ptr,                     // final cell state pointer   
        work_space_,                // workspace pointer
        workspace_byte_));          // workspace size    

    dim = output_dim;
    return dstData;
}

}}
