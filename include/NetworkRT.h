#ifndef NETWORKRT_H
#define NETWORKRT_H

#include "utils.h"
#include "Network.h"
#include "Layer.h"
#include "NvInfer.h"

namespace tkDNN {

class NetworkRT {

public:
    nvinfer1::DataType dtRT;
    nvinfer1::IBuilder *builderRT;
    nvinfer1::INetworkDefinition *networkRT; 
    
    nvinfer1::ICudaEngine *engineRT;
    nvinfer1::IExecutionContext *contextRT;
    void* buffersRT[2];
    int buf_input_idx, buf_output_idx;

    dataDim_t output_dim;
    dnnType *output;
    cudaStream_t stream;

    NetworkRT(Network *net);
    virtual ~NetworkRT();

    /**
        Do inferece
    */
    dnnType* infer(dataDim_t &dim, dnnType* data);

    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Layer *l);
    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Conv2d *l);
    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Activation *l);
    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Dense *l);
    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Pooling *l);
    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Softmax *l);
    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Route *l);
    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Reorg *l);
    nvinfer1::ITensor* convert_layer(nvinfer1::ITensor *input, Region *l);

};


}
#endif //NETWORKRT_H