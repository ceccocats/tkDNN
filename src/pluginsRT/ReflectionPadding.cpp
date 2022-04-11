#include <tkDNN/pluginsRT/ReflectionPadding.h>
using namespace nvinfer1;

std::vector<PluginField> ReflectionPaddingRTPluginCreator::mPluginAttributes;
PluginFieldCollection ReflectionPaddingRTPluginCreator::mFC{};

static const char* REFLECTIONPADDINGRT_PLUGIN_VERSION{"1"};
static const char* REFLECTIONPADDINGRT_PLUGIN_NAME{"ReflectionPaddingRT_tkDNN"};

ReflectionPaddingRT::ReflectionPaddingRT(int32_t padH, int32_t padW, int32_t input_h, int32_t input_w, int32_t output_h,
                                         int32_t output_w, int32_t c, int32_t n) {
    this->padH = padH;
    this->padW = padW;
    this->input_h  = input_h;
    this->input_w = input_w;
    this->output_h = output_h;
    this->output_w = output_w;
    this->n = n;
    this->c = c;
}

ReflectionPaddingRT::ReflectionPaddingRT(const void *data, size_t length) {
    const char* buf = reinterpret_cast<const char*>(data),*bufcheck=buf;
    padH = readBUF<int32_t>(buf);
    padW = readBUF<int32_t>(buf);
    input_h = readBUF<int32_t>(buf);
    input_w = readBUF<int32_t>(buf);
    output_h = readBUF<int32_t>(buf);
    output_w = readBUF<int32_t>(buf);
    n = readBUF<int32_t>(buf);
    c = readBUF<int32_t>(buf);
    assert(buf = bufcheck + length);
}

ReflectionPaddingRT::~ReflectionPaddingRT() {}

int ReflectionPaddingRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ReflectionPaddingRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{c,output_h,output_w};
}

int ReflectionPaddingRT::initialize() NOEXCEPT {
    return 0;
}

void ReflectionPaddingRT::terminate() NOEXCEPT {

}

size_t ReflectionPaddingRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int ReflectionPaddingRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                                 cudaStream_t stream) NOEXCEPT {
    dnnType* srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType* dstData = reinterpret_cast<dnnType*>(outputs[0]);
    reflection_pad2d_out_forward(padH,padW,srcData,dstData,input_h,input_w,c,n,stream);
    return 0;
}

#elif NV_TENSORRT_MAJOR <= 7
int32_t ReflectionPaddingRT::enqueue (int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream){
    dnnType* srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType* dstData = reinterpret_cast<dnnType*>(outputs[0]);
    reflection_pad2d_out_forward(padH,padW,srcData,dstData,input_h,input_w,c,n,stream);
    return 0;
}
#endif


size_t ReflectionPaddingRT::getSerializationSize() const NOEXCEPT {
    return 8*sizeof(int32_t);
}

void ReflectionPaddingRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf,padH);
    writeBUF(buf,padW);
    writeBUF(buf,input_h);
    writeBUF(buf,input_w);
    writeBUF(buf,output_h);
    writeBUF(buf,output_w);
    writeBUF(buf,n);
    writeBUF(buf,c);
}

void ReflectionPaddingRT::destroy() NOEXCEPT {
    delete this;
}

const char *ReflectionPaddingRT::getPluginType() const NOEXCEPT {
    return REFLECTIONPADDINGRT_PLUGIN_NAME;
}

const char *ReflectionPaddingRT::getPluginVersion() const NOEXCEPT {
    return REFLECTIONPADDINGRT_PLUGIN_VERSION;
}

const char *ReflectionPaddingRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void ReflectionPaddingRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2Ext *ReflectionPaddingRT::clone() const NOEXCEPT {
    auto *p = new ReflectionPaddingRT(padH,padW,input_h,input_w,output_h,output_w,c,n);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

DataType
ReflectionPaddingRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void ReflectionPaddingRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                          IGpuAllocator *gpuAllocator) NOEXCEPT {
}

bool ReflectionPaddingRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                       int nbInputs) const NOEXCEPT {
    return false;
}

bool ReflectionPaddingRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void
ReflectionPaddingRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                                     const DataType *inputTypes, const DataType *outputTypes,
                                     const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                                     PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT {

}

void ReflectionPaddingRT::detachFromContext() NOEXCEPT {

}

bool ReflectionPaddingRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}


ReflectionPaddingRTPluginCreator::ReflectionPaddingRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ReflectionPaddingRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ReflectionPaddingRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *ReflectionPaddingRTPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                                  size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ReflectionPaddingRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *
ReflectionPaddingRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    int padH = *(static_cast<const int32_t*>(fields[0].data));
    int padW = *(static_cast<const int32_t*>(fields[1].data));
    int inputH = *(static_cast<const int32_t*>(fields[2].data));
    int inputW = *(static_cast<const int32_t*>(fields[3].data));
    int outputH = *(static_cast<const int32_t*>(fields[4].data));
    int outputW = *(static_cast<const int32_t*>(fields[5].data));
    int n = *(static_cast<const int32_t*>(fields[6].data));
    int c = *(static_cast<const int32_t*>(fields[7].data));
    auto *pluginObj = new ReflectionPaddingRT(padH,padW,inputH,inputW,outputH,outputW,c,n);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ReflectionPaddingRTPluginCreator::getPluginName() const NOEXCEPT {
    return REFLECTIONPADDINGRT_PLUGIN_NAME;
}

const char *ReflectionPaddingRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return REFLECTIONPADDINGRT_PLUGIN_VERSION;
}

const PluginFieldCollection *ReflectionPaddingRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}


