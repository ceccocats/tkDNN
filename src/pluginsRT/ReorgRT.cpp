#include <tkDNN/pluginsRT/ReorgRT.h>
using namespace nvinfer1;

std::vector<PluginField> ReorgRTPluginCreator::mPluginAttributes;
PluginFieldCollection ReorgRTPluginCreator::mFC{};

ReorgRT::ReorgRT(int stride) {
    this->stride = stride;
}

ReorgRT::~ReorgRT() {}

ReorgRT::ReorgRT(const void *data, size_t length) {
    const char* buf  = reinterpret_cast<const char*>(data),*bufCheck = buf;
    stride = readBUF<int>(buf);
    c = readBUF<int>(buf);
    h = readBUF<int>(buf);
    w = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

int ReorgRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ReorgRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{inputs[0].d[0]*stride*stride, inputs[0].d[1]/stride, inputs[0].d[2]/stride};
}

void ReorgRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {
    c = inputDims[0].d[0];
    h = inputDims[0].d[1];
    w = inputDims[0].d[2];
}

int ReorgRT::initialize() NOEXCEPT {
    return 0;
}

void ReorgRT::terminate() NOEXCEPT {}

size_t ReorgRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int ReorgRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,cudaStream_t stream) NOEXCEPT {
    reorgForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
                 reinterpret_cast<dnnType*>(outputs[0]),
                 batchSize, c, h, w, stride, stream);
    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t ReorgRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    reorgForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
                 reinterpret_cast<dnnType*>(outputs[0]),
                 batchSize, c, h, w, stride, stream);
    return 0;
}
#endif


size_t ReorgRT::getSerializationSize() const NOEXCEPT {
    return 4*sizeof(int);
}

void ReorgRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, stride);
    writeBUF(buf, c);
    writeBUF(buf, h);
    writeBUF(buf, w);
    assert(buf == a + getSerializationSize());
}

bool ReorgRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
}

const char *ReorgRT::getPluginType() const NOEXCEPT {
    return "ReorgRT_tkDNN";
}

const char *ReorgRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

void ReorgRT::destroy() NOEXCEPT {
    delete this;
}

const char *ReorgRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void ReorgRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2 *ReorgRT::clone() const NOEXCEPT {
    auto *p = new ReorgRT(stride);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

ReorgRTPluginCreator::ReorgRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ReorgRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ReorgRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2 *ReorgRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ReorgRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *ReorgRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    assert(fc->nbFields == 1);
    assert(fields[0].type == PluginFieldType::kINT32);
    int stride = *(static_cast<const int *>(fields[0].data));
    auto *pluginObj = new ReorgRT(stride);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ReorgRTPluginCreator::getPluginName() const NOEXCEPT {
    return "ReorgRT_tkDNN";
}

const char *ReorgRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *ReorgRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}














