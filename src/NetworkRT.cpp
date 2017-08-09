#include <iostream>
#include <map>
#include "NvInfer.h"

#include "NetworkRT.h"

using namespace nvinfer1;
#include "pluginsRT/ActivationLeakyRT.cpp"
#include "pluginsRT/ReorgRT.cpp"
#include "pluginsRT/RegionRT.cpp"

// Logger for info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) override {
#ifdef DEBUG
        std::cout <<"TENSORRT LOG: "<< msg << std::endl;
#endif
    }
} loggerRT;

namespace tkDNN {

std::map<Layer*, nvinfer1::ITensor*>tensors; 

NetworkRT::NetworkRT(Network *net) {

    float rt_ver = float(NV_TENSORRT_MAJOR) + 
                   float(NV_TENSORRT_MINOR)/10 + 
                   float(NV_TENSORRT_PATCH)/100;
    std::cout<<"New NetworkRT (TensorRT v"<<rt_ver<<")\n";

    builderRT = createInferBuilder(loggerRT);
    networkRT = builderRT->createNetwork();
    dtRT = DataType::kFLOAT;

    //add input layer
    dataDim_t dim = net->layers[0]->input_dim;
    ITensor *input = networkRT->addInput("data", dtRT, 
                     DimsCHW{ dim.c, dim.h, dim.w});
    checkNULL(input);

    //add other layers
    for(int i=0; i<net->num_layers; i++) {
        Layer *l = net->layers[i];
        input = convert_layer(input, l);
        tensors[l] = input;
    }
    if(input == NULL)
        FatalError("conversion failed");
    output_dim = dim;
    Dims oDim = input->getDimensions();
    output_dim.c = oDim.d[0];
    output_dim.h = oDim.d[1];
    output_dim.w = oDim.d[2];

    //build tensorRT
    input->setName("out");
	networkRT->markOutput(*input);

	// Build the engine
	builderRT->setMaxBatchSize(1);
	builderRT->setMaxWorkspaceSize(1 << 20);

    std::cout<<"Building tensorRT cuda engine...\n";
	engineRT = builderRT->buildCudaEngine(*networkRT);
	// we don't need the network any more
	//networkRT->destroy();

    std::cout<<"create execution context\n";
	contextRT = engineRT->createExecutionContext();

	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	if(engineRT->getNbBindings() != 2)
        FatalError("Incorrect buffers number");
	
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	buf_input_idx = engineRT->getBindingIndex("data"); 
    buf_output_idx = engineRT->getBindingIndex("out");
    std::cout<<"input idex = "<<buf_input_idx<<" -> output index = "<<buf_output_idx<<"\n";

	// create GPU buffers and a stream
    checkCuda(cudaMalloc(&buffersRT[buf_input_idx],  dim.tot()*sizeof(value_type)));
    checkCuda(cudaMalloc(&buffersRT[buf_output_idx], output_dim.tot()*sizeof(value_type)));
    checkCuda(cudaMalloc(&output, output_dim.tot()*sizeof(value_type)));
	checkCuda(cudaStreamCreate(&stream));
}

NetworkRT::~NetworkRT() {

}

value_type* NetworkRT::infer(dataDim_t &dim, value_type* data) {

    checkCuda(cudaMemcpyAsync(buffersRT[buf_input_idx], data, dim.tot()*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    contextRT->enqueue(1, buffersRT, stream, nullptr);
    checkCuda(cudaMemcpyAsync(output, buffersRT[buf_output_idx], output_dim.tot()*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    cudaStreamSynchronize(stream);

    dim = output_dim;

    return output;
}

ITensor* NetworkRT::convert_layer(ITensor *input, Layer *l) {

    layerType_t type = l->getLayerType();

    if(type == LAYER_DENSE)
        return convert_layer(input, (Dense*) l);
    if(type == LAYER_CONV2D)
        return convert_layer(input, (Conv2d*) l);
    if(type == LAYER_POOLING)
        return convert_layer(input, (Pooling*) l);
    if(type == LAYER_ACTIVATION)
        return convert_layer(input, (Activation*) l);
    if(type == LAYER_SOFTMAX)
        return convert_layer(input, (Softmax*) l);
    if(type == LAYER_ROUTE)
        return convert_layer(input, (Route*) l);
    if(type == LAYER_REORG)
        return convert_layer(input, (Reorg*) l);
    if(type == LAYER_REGION)
        return convert_layer(input, (Region*) l);

    FatalError("Layer not implemented in tensorRT");
    return NULL;
}

ITensor* NetworkRT::convert_layer(ITensor *input, Dense *l) {
    //std::cout<<"convert Dense\n";

    Weights w { dtRT, l->data_h, l->inputs*l->outputs};
    Weights b = { dtRT, l->bias_h, l->outputs};
    IFullyConnectedLayer *lRT = networkRT->addFullyConnected(*input, l->outputs, w, b);

    checkNULL(lRT);
    return lRT->getOutput(0);
}

ITensor* NetworkRT::convert_layer(ITensor *input, Conv2d *l) {
    //std::cout<<"convert conv2D\n";

    Weights w { dtRT, l->data_h, l->inputs*l->outputs*l->kernelH*l->kernelW};
    Weights b;
    if(!l->batchnorm)
        b = { dtRT, l->bias_h, l->outputs};
    else
        b = { dtRT, nullptr, 0}; //on batchnorm bias are added later

    // Add a convolution layer with 20 outputs and a 5x5 filter.
    IConvolutionLayer *lRT = networkRT->addConvolution(*input, 
               l->outputs, DimsHW{l->kernelH, l->kernelW}, w, b);
    checkNULL(lRT);

    lRT->setStride(DimsHW{l->strideH, l->strideW});
    lRT->setPadding(DimsHW{l->paddingH, l->paddingW});

    if(l->batchnorm) {
        float eps = CUDNN_BN_MIN_EPSILON;

        //make power array of ones
        value_type *power_h = new value_type[l->outputs];
        for(int i=0; i<l->outputs; i++) power_h[i] = 1.0f;

        //convert mean
        for(int i=0; i<l->outputs; i++)
            l->mean_h[i] = l->mean_h[i] / -sqrt(eps + l->variance_h[i]); 
        
        //convert variance
        for(int i=0; i<l->outputs; i++)
            l->variance_h[i] = 1.0f / sqrt(eps + l->variance_h[i]); 

        Weights power{dtRT, power_h, l->outputs};
        Weights shift{dtRT, l->mean_h, l->outputs};
        Weights scale{dtRT, l->variance_h, l->outputs};
        IScaleLayer *lRT2 = networkRT->addScale(*lRT->getOutput(0), ScaleMode::kCHANNEL, 
                    shift, scale, power);
        checkNULL(lRT2);

        Weights shift2{dtRT, l->bias_h, l->outputs};
        Weights scale2{dtRT, l->scales_h, l->outputs};
        IScaleLayer *lRT3 = networkRT->addScale(*lRT2->getOutput(0), ScaleMode::kCHANNEL, 
                    shift2, scale2, power);
        checkNULL(lRT3);

        return lRT3->getOutput(0);
    }

    return lRT->getOutput(0);
}

ITensor* NetworkRT::convert_layer(ITensor *input, Pooling *l) {
    //std::cout<<"convert Pooling\n";

    IPoolingLayer *lRT = networkRT->addPooling(*input, 
        PoolingType::kMAX, DimsHW{l->winH, l->winW});
    checkNULL(lRT);
    lRT->setStride(DimsHW{l->strideH, l->strideW});

    return lRT->getOutput(0);
}

ITensor* NetworkRT::convert_layer(ITensor *input, Activation *l) {
    //std::cout<<"convert Activation\n";

    if(l->act_mode == ACTIVATION_LEAKY) {
        //std::cout<<"New plugin LEAKY\n";
        IPlugin *plugin = new ActivationLeakyRT();
        IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT->getOutput(0);
    }

    IActivationLayer *lRT = networkRT->addActivation(*input, ActivationType::kRELU);
    checkNULL(lRT);
    return lRT->getOutput(0);
}

ITensor* NetworkRT::convert_layer(ITensor *input, Softmax *l) {
    //std::cout<<"convert softmax\n";

    ISoftMaxLayer *lRT = networkRT->addSoftMax(*input);
    checkNULL(lRT);

    return lRT->getOutput(0);
}

ITensor* NetworkRT::convert_layer(ITensor *input, Route *l) {
    //std::cout<<"convert route\n";

    ITensor *tens[256];
    for(int i=0; i<l->layers_n; i++)
        tens[i] = tensors[l->layers[i]];
    IConcatenationLayer *lRT = networkRT->addConcatenation(tens, l->layers_n);
    checkNULL(lRT);

    return lRT->getOutput(0);
}

ITensor* NetworkRT::convert_layer(ITensor *input, Reorg *l) {
    //std::cout<<"convert Reorg\n";

    //std::cout<<"New plugin REORG\n";
    IPlugin *plugin = new ReorgRT(l->stride);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT->getOutput(0);
}

ITensor* NetworkRT::convert_layer(ITensor *input, Region *l) {
    //std::cout<<"convert Region\n";

    //std::cout<<"New plugin REGION\n";
    IPlugin *plugin = new RegionRT(l->classes, l->coords, l->num, l->thresh);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT->getOutput(0);
}

}
