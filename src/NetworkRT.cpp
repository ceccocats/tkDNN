#include <iostream>
#include <map>
#include <errno.h>
#include <string.h> // memcpy
#include <stdlib.h>

#include "kernels.h"

#include "utils.h"
#include "NvInfer.h"

#include "NetworkRT.h"
#include "Int8Calibrator.h"


using namespace nvinfer1;

// Logger for info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) NOEXCEPT override {
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
#if NV_TENSORRT_MAJOR >= 5
    std::cout<<"DLAs: "<<builderRT->getNbDLACores()<<"\n";
#endif
    networkRT = builderRT->createNetworkV2(0U);
#if NV_TENSORRT_MAJOR >= 6                
        configRT = builderRT->createBuilderConfig();
#endif
    
    if(!fileExist(name)) {
#if NV_TENSORRT_MAJOR >= 6                
        // Calibrator life time needs to last until after the engine is built.
        std::unique_ptr<IInt8EntropyCalibrator> calibrator;

        configRT->setAvgTimingIterations(1);
        configRT->setMinTimingIterations(1);
        configRT->setMaxWorkspaceSize(1 << 30);
        configRT->setFlag(BuilderFlag::kDEBUG);
#endif
        //input and dataType
        dataDim_t dim = net->layers[0]->input_dim;
        dtRT = DataType::kFLOAT;

        builderRT->setMaxBatchSize(net->maxBatchSize);
        configRT->setMaxWorkspaceSize(1 << 30);

        if(net->fp16 && builderRT->platformHasFastFp16()) {
            dtRT = DataType::kHALF;
#if NV_TENSORRT_MAJOR >= 6
            configRT->setFlag(BuilderFlag::kFP16);
#endif
        }
#if NV_TENSORRT_MAJOR >= 5
        if(net->dla && builderRT->getNbDLACores() > 0) {
            dtRT = DataType::kHALF;
            configRT->setFlag(BuilderFlag::kFP16);
            configRT->setFlag(BuilderFlag::kGPU_FALLBACK);
            configRT->setDefaultDeviceType(DeviceType::kDLA);
            configRT->setDLACore(0);
        }
#endif
#if NV_TENSORRT_MAJOR >= 6                
        if(net->int8 && builderRT->platformHasFastInt8()){
            // dtRT = DataType::kINT8;
            // builderRT->setInt8Mode(true);
            configRT->setFlag(BuilderFlag::kINT8);
            BatchStream calibrationStream(dim, 1, 100,      //TODO: check if 100 images are sufficient to the calibration (or 4951) 
                                            net->fileImgList, net->fileLabelList);
            
            /* The calibTableFilePath contains the path+filename of the calibration table.
             * Each calibration table can be found in the corresponding network folder (../Test/*).
             * Each network is located in a folder with the same name as the network.
             * If the folder has a different name, the calibration table is saved in build/ folder.
             */
            std::string calib_table_name = net->networkName + "/" + net->networkNameRT.substr(0, net->networkNameRT.find('.')) + "-calibration.table";
            std::string calib_table_path = net->networkName;
            if(!fileExist((const char *)calib_table_path.c_str()))
                calib_table_name = "./" + net->networkNameRT.substr(0, net->networkNameRT.find('.')) + "-calibration.table";

            calibrator.reset(new Int8EntropyCalibrator(calibrationStream, 1, 
                                            calib_table_name, 
                                            "data"));
            configRT->setInt8Calibrator(calibrator.get());
        }
#endif
        
        // add input layer
        ITensor *input = networkRT->addInput("data", DataType::kFLOAT, 
                        Dims3{ dim.c, dim.h, dim.w});
        checkNULL(input);

        //add other layers
        for(int i=0; i<net->num_layers; i++) {
            Layer *l = net->layers[i];
            ILayer *Ilay = convert_layer(input, l);
#if NV_TENSORRT_MAJOR >= 6                
            if(net->int8 && builderRT->platformHasFastInt8())
            {
                Ilay->setPrecision(DataType::kINT8);
            }
#endif
            Ilay->setName( (l->getLayerName() + std::to_string(i)).c_str() );
            
            input = Ilay->getOutput(0);
            input->setName( (l->getLayerName() + std::to_string(i) + "_out").c_str() );
            
            if(l->final)
                networkRT->markOutput(*input);
            tensors[l] = input;
        }
        if(input == NULL)
            FatalError("conversion failed");

        //build tensorRT
        input->setName("out");
        networkRT->markOutput(*input);

        std::cout<<"Selected maxBatchSize: "<<builderRT->getMaxBatchSize()<<"\n";
        printCudaMemUsage();
        std::cout<<"Building tensorRT cuda engine...\n";
#if NV_TENSORRT_MAJOR >= 6                
        engineRT = builderRT->buildEngineWithConfig(*networkRT, *configRT);
#else 
        engineRT = builderRT->buildCudaEngine(*networkRT);
        //engineRT = std::shared_ptr<nvinfer1::ICudaEngine>(builderRT->buildCudaEngine(*networkRT));
#endif
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
    std::cout<<"input index = "<<buf_input_idx<<" -> output index = "<<buf_output_idx<<"\n";


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
        buffersDIM[i] = dataDim_t(1, dim.d[0], dim.d[1], dim.d[2]);
        std::cout<<"RtBuffer "<<i<<"   dim: "; buffersDIM[i].print();
        checkCuda(cudaMalloc(&buffersRT[i], engineRT->getMaxBatchSize()*dim.d[0]*dim.d[1]*dim.d[2]*sizeof(dnnType)));
    }
    checkCuda(cudaMalloc(&output, engineRT->getMaxBatchSize()*output_dim.tot()*sizeof(dnnType)));
	checkCuda(cudaStreamCreate(&stream));
}

NetworkRT::~NetworkRT() {

}

dnnType* NetworkRT::infer(dataDim_t &dim, dnnType* data) {
    int batches = dim.n;
    if(batches > getMaxBatchSize()) {
        FatalError("input batch size too large");
    }

    checkCuda(cudaMemcpyAsync(buffersRT[buf_input_idx], data, batches*input_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    contextRT->enqueue(batches, buffersRT, stream, nullptr);
    checkCuda(cudaMemcpyAsync(output, buffersRT[buf_output_idx], batches*output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    checkCuda(cudaStreamSynchronize(stream));

    dim = output_dim;
    dim.n = batches;

    return output;
}

void NetworkRT::enqueue(int batchSize) {
    contextRT->enqueue(batchSize, buffersRT, stream, nullptr);
}

ILayer* NetworkRT::convert_layer(ITensor *input, Layer *l) {

    layerType_t type = l->getLayerType();

    if(type == LAYER_DENSE)
        return convert_layer(input, (Dense*) l);
    if(type == LAYER_CONV2D || type == LAYER_DECONV2D)
        return convert_layer(input, (Conv2d*) l);
    if(type == LAYER_POOLING)
        return convert_layer(input, (Pooling*) l);
    if(type == LAYER_ACTIVATION || type == LAYER_ACTIVATION_CRELU || type == LAYER_ACTIVATION_LEAKY || type == LAYER_ACTIVATION_MISH || type == LAYER_ACTIVATION_LOGISTIC)
        return convert_layer(input, (Activation*) l);
    if(type == LAYER_SOFTMAX)
        return convert_layer(input, (Softmax*) l);
    if(type == LAYER_ROUTE)
        return convert_layer(input, (Route*) l);
    if(type == LAYER_FLATTEN)
        return convert_layer(input, (Flatten*) l);
    if(type == LAYER_RESHAPE)
        return convert_layer(input, (Reshape*) l);
    if(type == LAYER_RESIZE)
        return convert_layer(input, (Resize*) l);
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
    if(type == LAYER_DEFORMCONV2D)
        return convert_layer(input, (DeformConv2d*) l);

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
    // std::cout<<"convert conv2D\n";
    // printf("%d %d %d %d %d\n", l->kernelH, l->kernelW, l->inputs, l->outputs, l->batchnorm);


    void *data_b, *bias_b, *bias2_b, *power_b, *mean_b, *variance_b, *scales_b;
    if(dtRT == DataType::kHALF) {
        data_b     = l->data16_h;    
        bias_b     = l->bias16_h;
        bias2_b    = l->bias216_h;
        power_b    = l->power16_h;
        mean_b     = l->mean16_h;
        variance_b = l->variance16_h;
        scales_b   = l->scales16_h;
    } else {
        data_b     = l->data_h;    
        bias_b     = l->bias_h;
        bias2_b    = l->bias2_h;
        power_b    = l->power_h;
        mean_b     = l->mean_h;
        variance_b = l->variance_h;
        scales_b   = l->scales_h;
    }


    Weights w { dtRT, data_b, l->inputs*l->outputs*l->kernelH*l->kernelW};
    Weights b;
    if(!l->batchnorm)
        b = { dtRT, bias_b, l->outputs};
    else{
        if (l->additional_bias)
            b = { dtRT, bias2_b, l->outputs}; 
        else
            b = { dtRT, nullptr, 0}; //on batchnorm bias are added later
    }

    ILayer *lRT = nullptr;
    if(!l->deConv) {
        IConvolutionLayer *lRTconv = networkRT->addConvolution(*input, 
            l->outputs, DimsHW{l->kernelH, l->kernelW}, w, b);
        checkNULL(lRTconv);
        lRTconv->setStride(DimsHW{l->strideH, l->strideW});
        lRTconv->setPadding(DimsHW{l->paddingH, l->paddingW});
        lRTconv->setNbGroups(l->groups);
        lRT = (ILayer*) lRTconv;
    } else {
        IDeconvolutionLayer *lRTconv = networkRT->addDeconvolution(*input, 
            l->outputs, DimsHW{l->kernelH, l->kernelW}, w, b);
        checkNULL(lRTconv);
        lRTconv->setStride(DimsHW{l->strideH, l->strideW});
        lRTconv->setPadding(DimsHW{l->paddingH, l->paddingW});
        lRTconv->setNbGroups(l->groups);
        lRT = (ILayer*) lRTconv;
        
        Dims d = lRTconv->getOutput(0)->getDimensions();
        //std::cout<<"DECONV: "<<d.d[0]<<" "<<d.d[1]<<" "<<d.d[2]<<" "<<d.d[3]<<"\n";
    }

    checkNULL(lRT);
    if(l->batchnorm) {
        Weights power{dtRT, power_b, l->outputs};
        Weights shift{dtRT, mean_b, l->outputs};
        Weights scale{dtRT, variance_b, l->outputs};
        // std::cout<<lRT->getNbOutputs()<<std::endl;
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
    // std::cout<<"convert Pooling\n";

    PoolingType ptype;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_MAX) ptype = PoolingType::kMAX;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_AVERAGE) ptype = PoolingType::kAVERAGE;
    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_AVERAGE_EXCLUDE_PADDING) ptype = PoolingType::kMAX_AVERAGE_BLEND;

    if(l->pool_mode == tkdnnPoolingMode_t::POOLING_MAX_FIXEDSIZE)
    {
        auto creator = getPluginRegistry()->getPluginCreator("MaxPoolingFixedSizeRT_tkDNN","1");
        std::vector<PluginField> mPluginAttributes;
        PluginFieldCollection mFC{};
        mPluginAttributes.emplace_back(PluginField("c",&l->output_dim.c,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("h",&l->output_dim.h,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("w",&l->output_dim.w,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("n",&l->output_dim.n,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("strideH",&l->strideH,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("strideW",&l->strideW,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("winSize",&l->winH,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("padding",&l->padding,PluginFieldType::kINT32,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
        auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
        auto *lRT = networkRT->addPluginV2(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
    }
    else
    {
        IPoolingLayer *lRT = networkRT->addPooling(*input, ptype, DimsHW{l->winH, l->winW});
        checkNULL(lRT);

        lRT->setPadding(DimsHW{l->paddingH, l->paddingW});
        lRT->setStride(DimsHW{l->strideH, l->strideW});
        return lRT;
    }  
}

ILayer* NetworkRT::convert_layer(ITensor *input, Activation *l) {
    //std::cout<<"convert Activation\n";

    if(l->act_mode == ACTIVATION_LEAKY) {
        //std::cout<<"New plugin LEAKY\n";
        
#if NV_TENSORRT_MAJOR < 6                
        // plugin version
        IPlugin *plugin = new ActivationLeakyRT(l->slope);
        IPluginLayer *lRT = networkRT->addPlugin(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
#else 
        IActivationLayer *lRT = networkRT->addActivation(*input, ActivationType::kLEAKY_RELU);
        lRT->setAlpha(l->slope);
        checkNULL(lRT);
        return lRT;
#endif

    } else if(l->act_mode == CUDNN_ACTIVATION_RELU) {
        IActivationLayer *lRT = networkRT->addActivation(*input, ActivationType::kRELU);
        checkNULL(lRT);
        return lRT;
    } else if(l->act_mode == CUDNN_ACTIVATION_SIGMOID) {
        IActivationLayer *lRT = networkRT->addActivation(*input, ActivationType::kSIGMOID);
        checkNULL(lRT);
        return lRT;
    }
    else if(l->act_mode == CUDNN_ACTIVATION_CLIPPED_RELU) {
        IPluginV2 *plugin = new ActivationReLUCeiling(l->ceiling);
        IPluginV2Layer *lRT = networkRT->addPluginV2(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
    } 
    else if(l->act_mode == ACTIVATION_MISH) {
        IPluginV2 *plugin = new ActivationMishRT();
        IPluginV2Layer *lRT = networkRT->addPluginV2(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
    }
    else if(l->act_mode == ACTIVATION_LOGISTIC) {
        IPluginV2 *plugin = new ActivationLogisticRT();
        IPluginV2Layer *lRT = networkRT->addPluginV2(&input, 1, *plugin);
        checkNULL(lRT);
        return lRT;
    }
    else {
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
    // std::cout<<"convert route\n";
    


    ITensor **tens = new ITensor*[l->layers_n];
    for(int i=0; i<l->layers_n; i++) {
        tens[i] = tensors[l->layers[i]];
        // for(int j=0; j<tens[i]->getDimensions().nbDims; j++) {
        //     std::cout<<tens[i]->getDimensions().d[j]<<" ";
        // }
        // std::cout<<"\n";
    }

    if(l->groups > 1){
        IPluginV2 *plugin = new RouteRT(l->groups, l->group_id);
        IPluginV2Layer *lRT = networkRT->addPluginV2(tens, l->layers_n, *plugin);
        checkNULL(lRT);
        return lRT;
    }
    IConcatenationLayer *lRT = networkRT->addConcatenation(tens, l->layers_n);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Flatten *l) {
    auto creator = getPluginRegistry()->getPluginCreator("FlattenConcatRT_tkDNN","1");
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC{};
    mPluginAttributes.emplace_back(PluginField("c",&l->c,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("h",&l->h,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("w",&l->w,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("rows",&l->rows,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("cols",&l->cols,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();

    auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
    auto *lRT = networkRT->addPluginV2(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Reshape *l) {
    // std::cout<<"convert Reshape\n";
    auto creator = getPluginRegistry()->getPluginCreator("ReshapeRT_tkDNN","1");
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC{};
    mPluginAttributes.emplace_back(PluginField("n",&l->n,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("c",&l->c,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("h",&l->h,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("w",&l->w,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
    auto *lRT = networkRT->addPluginV2(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Resize *l) {
    // std::cout<<"convert Resize\n";

    IResizeLayer *lRT = networkRT->addResize(*input); //default is kNEAREST
    checkNULL(lRT);
    Dims d{};
    lRT->setResizeMode(ResizeMode(l->mode));
    lRT->setOutputDimensions(Dims3{l->output_dim.c, l->output_dim.h, l->output_dim.w});
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Reorg *l) {
    //std::cout<<"convert Reorg\n";

    //std::cout<<"New plugin REORG\n";
    auto creator = getPluginRegistry()->getPluginCreator("ReorgRT_tkDNN","1");
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC{};
    mPluginAttributes.emplace_back(PluginField("stride",&l->stride,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("c",&l->input_dim.c,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("h",&l->input_dim.h,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("w",&l->input_dim.w,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
    auto *lRT = networkRT->addPluginV2(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Region *l) {
    //std::cout<<"convert Region\n";

    //std::cout<<"New plugin REGION\n";
    auto creator = getPluginRegistry()->getPluginCreator("RegionRT_tkDNN","1");
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC{};
    mPluginAttributes.emplace_back(PluginField("classes",&l->classes,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("coords",&l->coords,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("nums",&l->num,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("c",&l->input_dim.c,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("h",&l->input_dim.h,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("w",&l->input_dim.w,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
    auto *lRT = networkRT->addPluginV2(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Shortcut *l) {
    //std::cout<<"convert Shortcut\n";

    //std::cout<<"New plugin Shortcut\n";
    
    ITensor *back_tens = tensors[l->backLayer];

    if(l->backLayer->output_dim.c == l->output_dim.c && !l->mul) 
    {
        IElementWiseLayer *lRT = networkRT->addElementWise(*input, *back_tens, ElementWiseOperation::kSUM);
        checkNULL(lRT);
        return lRT;
    }
    else
    {
        // plugin version
        auto creator = getPluginRegistry()->getPluginCreator("ShortcutRT_tkDNN","1");
        std::vector<PluginField> mPluginAttributes;
        PluginFieldCollection mFC{};
        mPluginAttributes.emplace_back(PluginField("bc",&l->backLayer->output_dim.c,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("bh",&l->backLayer->output_dim.h,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("bw",&l->backLayer->output_dim.w,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("mul",&l->mul,PluginFieldType::kUNKNOWN,1));
        mPluginAttributes.emplace_back(PluginField("c",&l->c,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("h",&l->h,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("w",&l->w,PluginFieldType::kINT32,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
        auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
        auto **inputs = new ITensor*[2];
        inputs[0] = input;
        inputs[1] = back_tens; 
        auto *lRT = networkRT->addPluginV2(inputs, 2, *plugin);
        checkNULL(lRT);
        return lRT;
    }
}

ILayer* NetworkRT::convert_layer(ITensor *input, Yolo *l) {

    std::vector<dnnType> mask_h(l->mask_h,l->mask_h+sizeof(dnnType)*l->n_masks);
    std::vector<dnnType> bias_h(l->bias_h,l->bias_h+sizeof(dnnType)*2*l->n_masks*l->num);
    auto creator = getPluginRegistry()->getPluginCreator("YoloRT_tkDNN","1");
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC{};
    mPluginAttributes.emplace_back(PluginField("classes",&l->classes,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("num",&l->num,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("c",&l->input_dim.c,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("h",&l->input_dim.h,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("w",&l->input_dim.w,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("classNames",&l->classesNames[0],PluginFieldType::kUNKNOWN,l->classesNames.size()));
    mPluginAttributes.emplace_back(PluginField("mask_v",&mask_h[0],PluginFieldType::kFLOAT32,mask_h.size()));
    mPluginAttributes.emplace_back(PluginField("bias_v",&bias_h[0],PluginFieldType::kFLOAT32,bias_h.size()));
    mPluginAttributes.emplace_back(PluginField("n_masks",&l->n_masks,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("scale_xy",&l->scaleXY,PluginFieldType::kFLOAT32,1));
    mPluginAttributes.emplace_back(PluginField("nms_thresh",&l->nms_thresh,PluginFieldType::kFLOAT32,1));
    mPluginAttributes.emplace_back(PluginField("nms_kins",&l->nsm_kind,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("new_coords",&l->new_coords,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
    auto *lRT = networkRT->addPluginV2(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, Upsample *l) {
    //std::cout<<"convert Upsample\n";

    auto creator = getPluginRegistry()->getPluginCreator("UpSample_tkDNN","1");
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC{};
    mPluginAttributes.emplace_back(PluginField("stride",&l->stride,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("c",&l->c,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("h",&l->h,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("w",&l->w,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
    auto *lRT = networkRT->addPluginV2(&input, 1, *plugin);
    checkNULL(lRT);
    return lRT;
}

ILayer* NetworkRT::convert_layer(ITensor *input, DeformConv2d *l) {
    //std::cout<<"convert DEFORMABLE\n";
    ILayer *preconv = convert_layer(input, l->preconv);
    checkNULL(preconv);

    ITensor **inputs = new ITensor*[2];
    inputs[0] = input;
    inputs[1] = preconv->getOutput(0);

    //std::cout<<"New plugin DEFORMABLE\n";
    int height_ones = (l->input_dim.h + 2 * l->paddingH - (1 * (l->kernelH - 1) + 1)) / l->strideH + 1;
    int width_ones = (l->input_dim.w + 2 * l->paddingW - (1 * (l->kernelW - 1) + 1)) / l->strideW + 1;
    int dim_ones = l->input_dim.c * l->kernelH * l->kernelW * 1 * height_ones * width_ones;
    std::vector<dnnType> offsetV(2*l->chunk_dim);
    std::vector<dnnType> maskV(l->chunk_dim);
    std::vector<dnnType> dataV(l->input_dim.c*l->output_dim.c*l->kernelW*l->kernelH*1);
    std::vector<dnnType> bias2DV(l->output_dim.c);
    std::vector<dnnType> onesD1V(height_ones*width_ones);
    std::vector<dnnType> onesD2V(dim_ones);
    checkCuda(cudaMemcpy(offsetV.data(),l->offset,offsetV.size()*sizeof(dnnType),cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(maskV.data(),l->mask,sizeof(dnnType)*maskV.size(),cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(dataV.data(),l->data_d,sizeof(dnnType)*dataV.size(),cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(bias2DV.data(),l->bias2_d,sizeof(dnnType)*bias2DV.size(),cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(onesD1V.data(),l->ones_d1,sizeof(dnnType)*onesD1V.size(),cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(onesD2V.data(),l->ones_d2,sizeof(dnnType)*onesD2V.size(),cudaMemcpyDeviceToHost));
    auto creator = getPluginRegistry()->getPluginCreator("DeformableConvRT_tkDNN","1");
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFC{};
    mPluginAttributes.emplace_back(PluginField("chunk_dum",&l->chunk_dim,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("kh",&l->kernelH,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("kw",&l->kernelW,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("sh",&l->strideH,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("sw",&l->strideW,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("ph",&l->paddingH,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("pw",&l->paddingW,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("deformable_group",&l->deformableGroup,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("i_n",&l->input_dim.n,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("i_c",&l->input_dim.c,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("i_h",&l->input_dim.h,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("i_w",&l->input_dim.w,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("o_n",&l->output_dim.n,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("o_c",&l->output_dim.c,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("o_h",&l->output_dim.h,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("o_w",&l->output_dim.w,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("mask_v",&maskV[0],PluginFieldType::kFLOAT32,maskV.size()));
    mPluginAttributes.emplace_back(PluginField("offset_v",&offsetV[0],PluginFieldType::kFLOAT32,offsetV.size()));
    mPluginAttributes.emplace_back(PluginField("ones_d2_v",&onesD2V[0],PluginFieldType::kFLOAT32,onesD2V.size()));
    mPluginAttributes.emplace_back(PluginField("ones_d1_v",&onesD1V[0],PluginFieldType::kFLOAT32,onesD1V.size()));
    mPluginAttributes.emplace_back(PluginField("data_d_v",&dataV[0],PluginFieldType::kFLOAT32,dataV.size()));
    mPluginAttributes.emplace_back(PluginField("bias2_d_v",&bias2DV[0],PluginFieldType::kFLOAT32,bias2DV.size()));
    mPluginAttributes.emplace_back(PluginField("height_ones",&height_ones,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("width_ones",&width_ones,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("dim_ones",&dim_ones,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    auto *plugin = creator->createPlugin(l->getLayerName().c_str(),&mFC);
    auto *lRT = networkRT->addPluginV2(inputs, 2, *plugin);
    checkNULL(lRT);
    lRT->setName( ("Deformable" + std::to_string(l->id)).c_str() );
    delete[](inputs);
    // batchnorm
    void *bias_b, *power_b, *mean_b, *variance_b, *scales_b;
    if(dtRT == DataType::kHALF) {
        bias_b     = l->bias16_h;
        power_b    = l->power16_h;
        mean_b     = l->mean16_h;
        variance_b = l->variance16_h;
        scales_b   = l->scales16_h;
    } else {
        bias_b     = l->bias_h;
        power_b    = l->power_h;
        mean_b     = l->mean_h;
        variance_b = l->variance_h;
        scales_b   = l->scales_h;
    }

    Weights power{dtRT, power_b, l->outputs};
    Weights shift{dtRT, mean_b, l->outputs};
    Weights scale{dtRT, variance_b, l->outputs};
    //std::cout<<lRT->getNbOutputs()<<std::endl;
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

bool NetworkRT::serialize(const char *filename) {

    std::ofstream p(filename, std::ios::binary);
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

    runtimeRT = createInferRuntime(loggerRT);
    engineRT = runtimeRT->deserializeCudaEngine(gieModelStream, size);
    std::cout<<size<<std::endl;
    //if (gieModelStream) delete [] gieModelStream;

    return true;
}

void NetworkRT::destroy() {
    contextRT->destroy();
    engineRT->destroy();
    builderRT->destroy();
}

}}
