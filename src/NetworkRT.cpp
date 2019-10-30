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
    //std::cout<<"DLAs: "<<builderRT->getNbDLACores()<<"\n";
    networkRT = builderRT->createNetwork();

    if(!fileExist(name)) {

        //input and dataType
        dataDim_t dim = net->layers[0]->input_dim;
        dtRT = DataType::kFLOAT;

        builderRT->setMaxBatchSize(1);
        builderRT->setMaxWorkspaceSize(1 << 30);

        if(net->fp16 && builderRT->platformHasFastFp16()) {
            dtRT = DataType::kHALF;
            builderRT->setHalf2Mode(true);
        }
        /*
        if(net->dla && builderRT->getNbDLACores() > 0) {
            dtRT = DataType::kHALF;
            builderRT->setFp16Mode(true);
            builderRT->allowGPUFallback(true);
            builderRT->setDefaultDeviceType(DeviceType::kDLA);
            builderRT->setDLACore(0);
        }
        */
       
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
            input->setName( (l->getLayerName() + std::to_string(i) + "_out").c_str() );
            
            if(l->getLayerType() == LAYER_YOLO)
                networkRT->markOutput(*input);
            tensors[l] = input;
        }
        if(input == NULL)
            FatalError("conversion failed");

        //build tensorRT
        input->setName("out");
        networkRT->markOutput(*input);

        std::cout<<"Building tensorRT cuda engine...\n";
        engineRT = builderRT->buildCudaEngine(*networkRT);
        if(engineRT == nullptr)
            FatalError("cloud not build cuda engine")
        // we don't need the network any more
        //networkRT->destroy();
        std::cout<<"serialize net\n";
        serialize(name);
    } else {
        deserialize(name);
    }

    std::cout<<"create execution context\n";
	contextRT = engineRT->createExecutionContext();

	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	std::cout<<"Input/outputs numbers: "<<engineRT->getNbBindings()<<"\n";
    if(engineRT->getNbBindings() > MAX_BUFFERS_RT)
        FatalError("over RT buffer array size");

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
    output_dim.print();
	
    // create GPU buffers and a stream
    for(int i=0; i<engineRT->getNbBindings(); i++) {
        Dims dim = engineRT->getBindingDimensions(i);
        checkCuda(cudaMalloc(&buffersRT[i], dim.d[0]*dim.d[1]*dim.d[2]*sizeof(dnnType)));
    }
    checkCuda(cudaMalloc(&output, output_dim.tot()*sizeof(dnnType)));
	checkCuda(cudaStreamCreate(&stream));
}

NetworkRT::~NetworkRT() {

}

dnnType* NetworkRT::infer(dataDim_t &dim, dnnType* data) {

    checkCuda(cudaMemcpyAsync(buffersRT[buf_input_idx], data, input_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    contextRT->enqueue(1, buffersRT, stream, nullptr);
    checkCuda(cudaMemcpyAsync(output, buffersRT[buf_output_idx], output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
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
    if(type == LAYER_CONV2D || type == LAYER_DECONV2D)
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
    if(type == LAYER_YOLO)
        return convert_layer(input, (Yolo*) l);
    if(type == LAYER_UPSAMPLE)
        return convert_layer(input, (Upsample*) l);

    std::cout<<l->getLayerName()<<"\n";
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

    ILayer *lRT = nullptr;
    if(!l->deConv) {
        IConvolutionLayer *lRTconv = networkRT->addConvolution(*input, 
            l->outputs, DimsHW{l->kernelH, l->kernelW}, w, b);
        checkNULL(lRTconv);
        lRTconv->setStride(DimsHW{l->strideH, l->strideW});
        lRTconv->setPadding(DimsHW{l->paddingH, l->paddingW});
        lRT = (ILayer*) lRTconv;
    } else {
        IDeconvolutionLayer *lRTconv = networkRT->addDeconvolution(*input, 
            l->outputs, DimsHW{l->kernelH, l->kernelW}, w, b);
        checkNULL(lRTconv);
        lRTconv->setStride(DimsHW{l->strideH, l->strideW});
        lRTconv->setPadding(DimsHW{l->paddingH, l->paddingW});
        lRT = (ILayer*) lRTconv;
    }

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

    PoolingType ptype;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_MAX) ptype = PoolingType::kMAX;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_AVERAGE) ptype = PoolingType::kAVERAGE;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_AVERAGE_EXCLUDE_PADDING) ptype = PoolingType::kMAX_AVERAGE_BLEND;

    IPoolingLayer *lRT = networkRT->addPooling(*input, 
        ptype, DimsHW{l->winH, l->winW});
    checkNULL(lRT);
    lRT->setStride(DimsHW{l->strideH, l->strideW});
    lRT->setPadding(DimsHW{l->paddingH, l->paddingW});

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

    ITensor **tens = new ITensor*[l->layers_n];
    for(int i=0; i<l->layers_n; i++) {
        tens[i] = tensors[l->layers[i]];
    }
    IConcatenationLayer *lRT = networkRT->addConcatenation(tens, l->layers_n);
    //IPlugin *plugin = new RouteRT();
    //IPluginLayer *lRT = networkRT->addPlugin(tens, l->layers_n, *plugin);
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
    ITensor *back_tens = tensors[l->backLayer];
    IPlugin *plugin = new ShortcutRT();

    ITensor **inputs = new ITensor*[2];
    inputs[0] = input;
    inputs[1] = back_tens; 
    IPluginLayer *lRT = networkRT->addPlugin(inputs, 2, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Yolo *l) {
    //std::cout<<"convert Yolo\n";

    //std::cout<<"New plugin YOLO\n";
    IPlugin *plugin = new YoloRT(l->classes, l->num, l);
    IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Upsample *l) {
    //std::cout<<"convert Upsample\n";

    //std::cout<<"New plugin UPSAMPLE\n";
    IPlugin *plugin = new UpsampleRT(l->stride);
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

    pluginFactory = new PluginFactory();
    runtimeRT = createInferRuntime(loggerRT);
    engineRT = runtimeRT->deserializeCudaEngine(gieModelStream, size, (IPluginFactory *) pluginFactory);
    //if (gieModelStream) delete [] gieModelStream;

    return true;
}



IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
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

    if(name.find("Shortcut") == 0) {
        ShortcutRT *r = new ShortcutRT();
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        return r;
    } 

    if(name.find("Yolo") == 0) {
        YoloRT *r = new YoloRT(readBUF<int>(buf),    //classes
                                readBUF<int>(buf));   //num
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        for(int i=0; i<r->num; i++)
            r->mask[i] = readBUF<dnnType>(buf);
        for(int i=0; i<3*2*r->num; i++)
            r->bias[i] = readBUF<dnnType>(buf);

		// save classes names
        r->classesNames.resize(r->classes);
		for(int i=0; i<r->classes; i++) {
            char tmp[YOLORT_CLASSNAME_W];
			for(int j=0; j<YOLORT_CLASSNAME_W; j++)
				tmp[j] = readBUF<char>(buf);
            r->classesNames[i] = std::string(tmp);
		}

        yolos[n_yolos++] = r;
        return r;
    } 

    if(name.find("Upsample") == 0) {
        UpsampleRT *r = new UpsampleRT(readBUF<int>(buf)); //stride
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        return r;
    } 
/*
    if(name.find("Route") == 0) {
        RouteRT *r = new RouteRT();
        r->in = readBUF<int>(buf);
        for(int i=0; i<RouteRT::MAX_INPUTS; i++)
            r->c_in[i] = readBUF<int>(buf);
        r->c = readBUF<int>(buf);
        r->h = readBUF<int>(buf);
        r->w = readBUF<int>(buf);
        return r;
    } 
*/
    FatalError("Cant deserialize Plugin");
    return NULL;
}

}}
