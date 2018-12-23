#ifndef NETWORKRT_H
#define NETWORKRT_H

#include "utils.h"
#include "Network.h"
#include "Layer.h"
#include "NvInfer.h"

namespace tk { namespace dnn {

class NetworkRT {

public:
    nvinfer1::DataType dtRT;
    nvinfer1::IBuilder *builderRT;
    nvinfer1::IRuntime *runtimeRT;
    nvinfer1::INetworkDefinition *networkRT; 
    
    nvinfer1::ICudaEngine *engineRT;
    nvinfer1::IExecutionContext *contextRT;

    const static int MAX_BUFFERS_RT = 10;
    void* buffersRT[MAX_BUFFERS_RT];
    int buf_input_idx, buf_output_idx;

    dataDim_t input_dim, output_dim;
    dnnType *output;
    cudaStream_t stream;

    NetworkRT(Network *net, const char *name);
    virtual ~NetworkRT();

    /**
        Do inferece
    */
    dnnType* infer(dataDim_t &dim, dnnType* data);
    void enqueue();    

    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Layer *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Conv2d *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Activation *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Dense *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Pooling *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Softmax *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Route *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Reorg *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Region *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Shortcut *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Yolo *l);
    nvinfer1::ILayer* convert_layer(nvinfer1::ITensor *input, Upsample *l);

    bool serialize(const char *filename);
    bool deserialize(const char *filename);
};


template<typename T> void writeBUF(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T> T readBUF(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

}}
#endif //NETWORKRT_H
