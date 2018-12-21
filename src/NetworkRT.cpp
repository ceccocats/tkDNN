#include <iostream>
#include <map>
#include <errno.h>
#include <string.h> // memcpy
#include <stdlib.h>

#include "kernels.h"

#include "utils.h"
#include "NvInfer.h"
#include "NetworkRT.h"

using namespace nvinfer1;
#include "pluginsRT/ActivationLeakyRT.cpp"
#include "pluginsRT/ReorgRT.cpp"
#include "pluginsRT/RegionRT.cpp"
#include "pluginsRT/ShortcutRT.cpp"
#include "pluginsRT/Int8Calibrator.cpp"

// Logger for info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) override {
#ifdef DEBUG
        std::cout <<"TENSORRT LOG: "<< msg << std::endl;
#endif
    }
} loggerRT;

namespace tk { namespace dnn {

std::map<Layer*, nvinfer1::ITensor*>tensors; 

NetworkRT::NetworkRT(Network *net, const char *name) {

    float rt_ver = float(NV_TENSORRT_MAJOR) + 
                   float(NV_TENSORRT_MINOR)/10 + 
                   float(NV_TENSORRT_PATCH)/100;
    std::cout<<"New NetworkRT (TensorRT v"<<rt_ver<<")\n";
  
    builderRT = createInferBuilder(loggerRT);
    std::cout<<"Float16 support: "<<builderRT->platformHasFastFp16()<<"\n";
	std::cout<<"Int8 support: "<<builderRT->platformHasFastInt8()<<"\n";
    networkRT = builderRT->createNetwork();

    if(!fileExist(name)) {

        //input and dataType
        dataDim_t dim = net->layers[0]->input_dim;
        dtRT = DataType::kFLOAT;

        builderRT->setMaxBatchSize(1);
        builderRT->setMaxWorkspaceSize(1 << 30);

        //change datatype based on system specs
        if(builderRT->platformHasFastInt8()) {
            BatchStream bstream({32,dim.c, dim.h, dim.w}, 32, 1);
            Int8EntropyCalibrator calib(bstream, 0, false);
            builderRT->setInt8Mode(true);
            builderRT->setInt8Calibrator(&calib);

        } else if(net->fp16 && builderRT->platformHasFastFp16()) {
            dtRT = DataType::kHALF;
            builderRT->setHalf2Mode(true);
        }

        //add input layer
        ITensor *input = networkRT->addInput("data", DataType::kFLOAT, 
                        DimsCHW{ dim.c, dim.h, dim.w});
        checkNULL(input);

        //add other layers
        for(int i=0; i<net->num_layers; i++) {
            Layer *l = net->layers[i];
            ILayer *Ilay = convert_layer(input, l);
            Ilay->setName( (l->getLayerName() + std::to_string(i)).c_str() );
            
            input = Ilay->getOutput(0);
            tensors[l] = input;
        }
        if(input == NULL)
            FatalError("conversion failed");

        //build tensorRT
        input->setName("out");
        networkRT->markOutput(*input);

        std::cout<<"Building tensorRT cuda engine...\n";
        engineRT = builderRT->buildCudaEngine(*networkRT);
        // we don't need the network any more
        //networkRT->destroy();
        serialize(name);
    } else {
        deserialize(name);
    }

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


    Dims iDim = engineRT->getBindingDimensions(buf_input_idx);
    input_dim.n = 1;
    input_dim.c = iDim.d[0];
    input_dim.h = iDim.d[1];
    input_dim.w = iDim.d[2];
    input_dim.print();

    Dims oDim = engineRT->getBindingDimensions(buf_output_idx);
    output_dim.n = 1;
    output_dim.c = oDim.d[0];
    output_dim.h = oDim.d[1];
    output_dim.w = oDim.d[2];
	
    // create GPU buffers and a stream
    checkCuda(cudaMalloc(&buffersRT[buf_input_idx],  input_dim.tot()*sizeof(dnnType)));
    checkCuda(cudaMalloc(&buffersRT[buf_output_idx], output_dim.tot()*sizeof(dnnType)));
    checkCuda(cudaMalloc(&output, output_dim.tot()*sizeof(dnnType)));
	checkCuda(cudaStreamCreate(&stream));
}

NetworkRT::~NetworkRT() {

}

dnnType* NetworkRT::infer(dataDim_t &dim, dnnType* data) {

    checkCuda(cudaMemcpyAsync(buffersRT[buf_input_idx], data, input_dim.tot()*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    contextRT->enqueue(1, buffersRT, stream, nullptr);
    checkCuda(cudaMemcpyAsync(output, buffersRT[buf_output_idx], output_dim.tot()*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    cudaStreamSynchronize(stream);

    dim = output_dim;

    return output;
}

void NetworkRT::enqueue() {
    contextRT->enqueue(1, buffersRT, stream, nullptr);
}

ILayer* NetworkRT::convert_layer(ITensor *input, Layer *l) {

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
    if(type == LAYER_SHORTCUT)
        return convert_layer(input, (Shortcut*) l);

    FatalError("Layer not implemented in tensorRT");
    return NULL;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Dense *l) {
    //std::cout<<"convert Dense\n";
    void *data_b, *bias_b;
    if(dtRT == DataType::kHALF) {
        data_b     = l->data16_h;    
        bias_b     = l->bias16_h;
    } else {
        data_b     = l->data_h;    
        bias_b     = l->bias_h;
    }

    Weights w { dtRT, data_b, l->inputs*l->outputs};
    Weights b = { dtRT, bias_b, l->outputs};
    IFullyConnectedLayer *lRT = networkRT->addFullyConnected(*input, l->outputs, w, b);

    checkNULL(lRT);
    return lRT;
}


ILayer* NetworkRT::convert_layer(ITensor *input, Conv2d *l) {
    //std::cout<<"convert conv2D\n";

    void *data_b, *bias_b, *power_b, *mean_b, *variance_b, *scales_b;
    if(dtRT == DataType::kHALF) {
        data_b     = l->data16_h;    
        bias_b     = l->bias16_h;
        power_b    = l->power16_h;
        mean_b     = l->mean16_h;
        variance_b = l->variance16_h;
        scales_b   = l->scales16_h;
    } else {
        data_b     = l->data_h;    
        bias_b     = l->bias_h;
        power_b    = l->power_h;
        mean_b     = l->mean_h;
        variance_b = l->variance_h;
        scales_b   = l->scales_h;
    }


    Weights w { dtRT, data_b, l->inputs*l->outputs*l->kernelH*l->kernelW};
    Weights b;
    if(!l->batchnorm)
        b = { dtRT, bias_b, l->outputs};
    else
        b = { dtRT, nullptr, 0}; //on batchnorm bias are added later

    // Add a convolution layer with 20 outputs and a 5x5 filter.
    IConvolutionLayer *lRT = networkRT->addConvolution(*input, 
               l->outputs, DimsHW{l->kernelH, l->kernelW}, w, b);
    checkNULL(lRT);

    lRT->setStride(DimsHW{l->strideH, l->strideW});
    lRT->setPadding(DimsHW{l->paddingH, l->paddingW});

    if(l->batchnorm) {
        Weights power{dtRT, power_b, l->outputs};
        Weights shift{dtRT, mean_b, l->outputs};
        Weights scale{dtRT, variance_b, l->outputs};
        IScaleLayer *lRT2 = networkRT->addScale(*lRT->getOutput(0), ScaleMode::kCHANNEL, 
                    shift, scale, power);
        checkNULL(lRT2);

        Weights shift2{dtRT, bias_b, l->outputs};
        Weights scale2{dtRT, scales_b, l->outputs};
        IScaleLayer *lRT3 = networkRT->addScale(*lRT2->getOutput(0), ScaleMode::kCHANNEL, 
                    shift2, scale2, power);
        checkNULL(lRT3);

        return lRT3;
    }

    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Pooling *l) {
    //std::cout<<"convert Pooling\n";

    IPoolingLayer *lRT = networkRT->addPooling(*input, 
        PoolingType::kMAX, DimsHW{l->winH, l->winW});
    checkNULL(lRT);
    lRT->setStride(DimsHW{l->strideH, l->strideW});

    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Activation *l) {
    //std::cout<<"convert Activation\n";

    if(l->act_mode == ACTIVATION_LEAKY) {
        //std::cout<<"New plugin LEAKY\n";
        IPlugin *plugin = new ActivationLeakyRT();
        IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;

    } else if(l->act_mode == CUDNN_ACTIVATION_RELU) {
        IActivationLayer *lRT = networkRT->addActivation(*input, ActivationType::kRELU);
        checkNULL(lRT);
        return lRT;
    
    } else {
        FatalError("this Activation mode is not yet implemented");
        return NULL;
    }
}

ILayer* NetworkRT::convert_layer(ITensor *input, Softmax *l) {
    //std::cout<<"convert softmax\n";

    ISoftMaxLayer *lRT = networkRT->addSoftMax(*input);
    checkNULL(lRT);

    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Route *l) {
    //std::cout<<"convert route\n";

    ITensor *tens[256];
    for(int i=0; i<l->layers_n; i++)
        tens[i] = tensors[l->layers[i]];
    IConcatenationLayer *lRT = networkRT->addConcatenation(tens, l->layers_n);
    checkNULL(lRT);

    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Reorg *l) {
    //std::cout<<"convert Reorg\n";

    //std::cout<<"New plugin REORG\n";
    IPlugin *plugin = new ReorgRT(l->stride);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Region *l) {
    //std::cout<<"convert Region\n";

    //std::cout<<"New plugin REGION\n";
    IPlugin *plugin = new RegionRT(l->classes, l->coords, l->num);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Shortcut *l) {
    //std::cout<<"convert Shortcut\n";

    //std::cout<<"New plugin Shortcut\n";
    ITensor *tens = tensors[l->backLayer];
    IPlugin *plugin = new ShortcutRT(tens);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}


bool NetworkRT::serialize(const char *filename) {

    std::ofstream p(filename);
    if (!p) {
        FatalError("could not open plan output file");
        return false;
    }

    IHostMemory *ptr = engineRT->serialize();
    if(ptr == nullptr)
        FatalError("Cant serialize network");

    p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
    ptr->destroy();
    return true;
}

class PluginFactory : IPluginFactory
{
public:
	virtual IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
        const char * buf = reinterpret_cast<const char*>(serialData);
        
        std::string name(layerName);

        if(name.find("Activation") == 0) {
            ActivationLeakyRT *a = new ActivationLeakyRT();
            a->size = readBUF<int>(buf);
            return a;
        }

        if(name.find("Region") == 0) {
            RegionRT *r = new RegionRT(readBUF<int>(buf),    //classes
                                       readBUF<int>(buf),    //coords
                                       readBUF<int>(buf));   //num

        	r->c = readBUF<int>(buf);
		    r->h = readBUF<int>(buf);
		    r->w = readBUF<int>(buf);
            return r;
        } 

        if(name.find("Reorg") == 0) {
            ReorgRT *r = new ReorgRT(readBUF<int>(buf)); //stride
        	r->c = readBUF<int>(buf);
		    r->h = readBUF<int>(buf);
		    r->w = readBUF<int>(buf);
            return r;
        } 

        FatalError("Cant deserialize Plugin");
        return NULL;
    }
};

bool NetworkRT::deserialize(const char *filename) {

    char *gieModelStream{nullptr};
    size_t size{0};
    std::ifstream file(filename, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        gieModelStream = new char[size];
        file.read(gieModelStream, size);
        file.close();
    }

    PluginFactory plfact;

    runtimeRT = createInferRuntime(loggerRT);
    engineRT = runtimeRT->deserializeCudaEngine(gieModelStream, size, (IPluginFactory *) &plfact);
    //if (gieModelStream) delete [] gieModelStream;

    return true;
}

}}
