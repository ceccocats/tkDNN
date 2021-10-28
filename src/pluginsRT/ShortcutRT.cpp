#include <tkDNN/pluginsRT/ShortcutRT.h>
using namespace nvinfer1;

std::vector<PluginField> ShortcutRTPluginCreator::mPluginAttributes;
PluginFieldCollection ShortcutRTPluginCreator::mFC{};

static const char* SHORTCUTRT_PLUGIN_VERSION{"1"};
static const char* SHORTCUTRT_PLUGIN_NAME{"ShortcutRT_tkDNN"};

ShortcutRT::ShortcutRT(int bc,int bh,int bw,int c,int h,int w,bool mul) {
    this->bc = bc;
    this->bh = bh;
    this->bw = bw;
    this->mul = mul;
    this->c = c;
    this->h = h;
    this->w = w;
}

ShortcutRT::~ShortcutRT() {}

ShortcutRT::ShortcutRT(const void *data, size_t length) {
    const char* buf =reinterpret_cast<const char*>(data),*bufCheck = buf;
    bc = readBUF<int>(buf);
    bh = readBUF<int>(buf);
    bw = readBUF<int>(buf);
    mul = readBUF<bool>(buf);
    c = readBUF<int>(buf);
    h = readBUF<int>(buf);
    w = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

int ShortcutRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ShortcutRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
}

int ShortcutRT::initialize() NOEXCEPT {
    return 0;
}

void ShortcutRT::terminate() NOEXCEPT {}

size_t ShortcutRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT { return 0; }

#if NV_TENSORRT_MAJOR > 7
int ShortcutRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) NOEXCEPT {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *srcDataBack = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

    checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    shortcutForward(srcDataBack, dstData, batchSize, c, h, w, 1, batchSize, bc, bh, bw, 1, mul, stream);

    return 0;
}
#elif NV_TENSORRT_MAJOR <= 7
int32_t ShortcutRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                            cudaStream_t stream) {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *srcDataBack = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

    checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    shortcutForward(srcDataBack, dstData, batchSize, c, h, w, 1, batchSize, bc, bh, bw, 1, mul, stream);

    return 0;
}
#endif


size_t ShortcutRT::getSerializationSize() const NOEXCEPT {
    return 6*sizeof(int) + sizeof(bool);
}

void ShortcutRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, bc);
    writeBUF(buf, bh);
    writeBUF(buf, bw);
    writeBUF(buf, mul);
    writeBUF(buf, c);
    writeBUF(buf, h);
    writeBUF(buf, w);
    assert(buf == a + getSerializationSize());
}

bool ShortcutRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
}

const char *ShortcutRT::getPluginType() const NOEXCEPT {
    return SHORTCUTRT_PLUGIN_NAME;
}

const char *ShortcutRT::getPluginVersion() const NOEXCEPT {
    return SHORTCUTRT_PLUGIN_VERSION;
}

void ShortcutRT::destroy() NOEXCEPT {
    delete this;
}

const char *ShortcutRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void ShortcutRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2Ext *ShortcutRT::clone() const NOEXCEPT {
    auto *p = new ShortcutRT(bc,bh,bw,c,h,w,mul);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

void ShortcutRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                 const bool *outputIsBroadcast, PluginFormat floatFormat,
                                 int32_t maxBatchSize) NOEXCEPT {

}

bool ShortcutRT::isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool *inputIsBroadcasted,
                                              int32_t nbInputs) const NOEXCEPT {
    return false;
}

bool ShortcutRT::canBroadcastInputAcrossBatch(int32_t inputIndex) const NOEXCEPT {
    return false;
}

void ShortcutRT::attachToContext(cudnnContext *, cublasContext *, IGpuAllocator *) NOEXCEPT {

}

void ShortcutRT::detachFromContext() NOEXCEPT {

}

DataType ShortcutRT::getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes, int32_t nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}


ShortcutRTPluginCreator::ShortcutRTPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("bc", nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("bh", nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("bw", nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("mul", nullptr,PluginFieldType::kUNKNOWN,1));
    mPluginAttributes.emplace_back(PluginField("c", nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("h", nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("w", nullptr,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ShortcutRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ShortcutRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *ShortcutRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ShortcutRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *ShortcutRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    assert(fc->nbFields == 7);
    int bc = *(static_cast<const int *>(fields[0].data));
    int bh = *(static_cast<const int *>(fields[1].data));
    int bw = *(static_cast<const int *>(fields[2].data));
    bool mul = *(static_cast<const bool *>(fields[3].data));
    int c = *(static_cast<const int *>(fields[4].data));
    int h = *(static_cast<const int *>(fields[5].data));
    int w = *(static_cast<const int *>(fields[6].data));
    auto *pluginObj = new ShortcutRT(bc,bh,bw,c,h,w,mul);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ShortcutRTPluginCreator::getPluginName() const NOEXCEPT {
    return SHORTCUTRT_PLUGIN_NAME;
}

const char *ShortcutRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return SHORTCUTRT_PLUGIN_VERSION;
}

const PluginFieldCollection *ShortcutRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}














