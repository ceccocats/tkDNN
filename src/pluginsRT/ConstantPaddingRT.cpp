#include <tkDNN/pluginsRT/ConstantPaddingRT.h>

using namespace nvinfer1;

std::vector<PluginField> ConstantPaddingRTPluginCreator::mPluginAttributes;
PluginFieldCollection ConstantPaddingRTPluginCreator::mFC{};

static const char* CONSTANTPADDINGRT_PLUGIN_VERSION{"1"};
static const char* CONSTANTPADDINGRT_PLUGIN_NAME{"ConstantPaddingRT_tkDNN"};

ConstantPaddingRT::ConstantPaddingRT(int32_t padH, int32_t padW, int32_t n, int32_t c, int32_t i_h, int32_t i_w,
                                     int32_t o_h, int32_t o_w, float constant) {
    this->padH = padH;
    this->padW = padW;
    this->n = n;
    this->c = c;
    this->i_h = i_h;
    this->i_w = i_w;
    this->o_h = o_h;
    this->o_w = o_w;
    this->constant = constant;

}

ConstantPaddingRT::ConstantPaddingRT(const void *data, size_t length) {
    const char* buf = reinterpret_cast<const char*>(data),*bufcheck=buf;
    padH = readBUF<int32_t>(buf);
    padW = readBUF<int32_t>(buf);
    i_h = readBUF<int32_t>(buf);
    i_w = readBUF<int32_t>(buf);
    o_h = readBUF<int32_t>(buf);
    o_w = readBUF<int32_t>(buf);
    n = readBUF<int32_t>(buf);
    c = readBUF<int32_t>(buf);
    constant = readBUF<float>(buf);
    assert(buf = bufcheck + length);
}

ConstantPaddingRT::~ConstantPaddingRT() {}

int ConstantPaddingRT::getNbOutputs() const NOEXCEPT{
    return 1;
}

Dims ConstantPaddingRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{c,o_h,o_w};
}

int ConstantPaddingRT::initialize() NOEXCEPT {
    return 0;
}

void ConstantPaddingRT::terminate() NOEXCEPT {

}

size_t ConstantPaddingRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int ConstantPaddingRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) NOEXCEPT {
     dnnType* srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
     dnnType* dstData = reinterpret_cast<dnnType*>(outputs[0]);
     constant_pad2d_forward(srcData,dstData,i_h,i_w,o_h,o_w,c,n,padH,padW,constant,stream);
     return 0;
}
#elif NV_TENSORRT_MAJOR <= 7
    int32_t enqueue (int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
        dnnType* srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
        dnnType* dstData = reinterpret_cast<dnnType*>(outputs[0]);
        constant_pad2d_forward(srcData,dstData,i_h,i_w,o_h,o_w,c,n,padH,padW,constant,stream);
        return 0;
}
#endif

size_t ConstantPaddingRT::getSerializationSize() const NOEXCEPT {
    return (8*sizeof(int32_t) + 1*sizeof(float));
}

void ConstantPaddingRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf,padH);
    writeBUF(buf,padW);
    writeBUF(buf,i_h);
    writeBUF(buf,i_w);
    writeBUF(buf,o_h);
    writeBUF(buf,o_w);
    writeBUF(buf,n);
    writeBUF(buf,c);
    writeBUF(buf,constant);
}

void ConstantPaddingRT::destroy() NOEXCEPT {
    delete this;
}

const char* ConstantPaddingRT::getPluginType() const NOEXCEPT {
    return CONSTANTPADDINGRT_PLUGIN_NAME;
}

const char* ConstantPaddingRT::getPluginVersion() const NOEXCEPT {
    return CONSTANTPADDINGRT_PLUGIN_VERSION;
}

const char* ConstantPaddingRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void ConstantPaddingRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2Ext *ConstantPaddingRT::clone() const NOEXCEPT {
    auto *p = new ConstantPaddingRT(padH,padW,n,c,i_h,i_w,o_h,o_w,constant);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

DataType ConstantPaddingRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                              int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void ConstantPaddingRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                        IGpuAllocator *gpuAllocator) NOEXCEPT {

}

bool ConstantPaddingRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                     int nbInputs) const NOEXCEPT {
    return false;
}

bool ConstantPaddingRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void ConstantPaddingRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims,
                                        int32_t nbOutputs, const DataType *inputTypes, const DataType *outputTypes,
                                        const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                                        PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT {

}

void ConstantPaddingRT::detachFromContext() NOEXCEPT {

}

bool ConstantPaddingRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

ConstantPaddingRTPluginCreator::ConstantPaddingRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ConstantPaddingRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ConstantPaddingRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *ConstantPaddingRTPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                                size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ConstantPaddingRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *ConstantPaddingRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    int padH = *(static_cast<const int32_t*>(fields[0].data));
    int padW = *(static_cast<const int32_t*>(fields[1].data));
    int inputH = *(static_cast<const int32_t*>(fields[2].data));
    int inputW = *(static_cast<const int32_t*>(fields[3].data));
    int outputH = *(static_cast<const int32_t*>(fields[4].data));
    int outputW = *(static_cast<const int32_t*>(fields[5].data));
    int n = *(static_cast<const int32_t*>(fields[6].data));
    int c = *(static_cast<const int32_t*>(fields[7].data));
    float constant = *(static_cast<const float*>(fields[8].data));
    auto *pluginObj = new ConstantPaddingRT(padH,padW,n,c,inputH,inputW,outputH,outputW,constant);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ConstantPaddingRTPluginCreator::getPluginName() const NOEXCEPT {
    return CONSTANTPADDINGRT_PLUGIN_NAME;
}

const char *ConstantPaddingRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return CONSTANTPADDINGRT_PLUGIN_VERSION;
}

const PluginFieldCollection *ConstantPaddingRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}
