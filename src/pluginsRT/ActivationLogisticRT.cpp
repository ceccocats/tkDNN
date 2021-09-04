#include <tkDNN/pluginsRT/ActivationLogisticRT.h>
using namespace nvinfer1;
std::vector<PluginField> ActivationLogisticRTPluginCreator::mPluginAttributes;
PluginFieldCollection ActivationLogisticRTPluginCreator::mFC{};

ActivationLogisticRT::ActivationLogisticRT() {}

ActivationLogisticRT::ActivationLogisticRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char *>(data), *bufCheck = buf;
    size = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

ActivationLogisticRT::~ActivationLogisticRT() {}

int ActivationLogisticRT::getNbOutputs() const NOEXCEPT {
return 1;
}

Dims ActivationLogisticRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return inputs[0];
}

void ActivationLogisticRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims,
                                               int nbOutputs, DataType type, PluginFormat format,
                                               int maxBatchSize) NOEXCEPT {
    size = 1;
    for (int i = 0; i < outputDims[0].nbDims; i++)
        size *= outputDims[0].d[i];
}

int ActivationLogisticRT::initialize() NOEXCEPT {
    return 0;
}

void ActivationLogisticRT::terminate() NOEXCEPT {}

size_t ActivationLogisticRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

int ActivationLogisticRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                                  cudaStream_t stream) NOEXCEPT {
    activationLOGISTICForward((dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
                              reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, stream);
    return 0;
}

size_t ActivationLogisticRT::getSerializationSize() const NOEXCEPT {
    return 1 * sizeof(int);
}

void ActivationLogisticRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char *>(buffer);
    writeBUF(buf, size);
}

const char* ActivationLogisticRT::getPluginType() const NOEXCEPT {
    return "ActivationLogisticRT_tkDNN";
}

const char* ActivationLogisticRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

void ActivationLogisticRT::destroy() NOEXCEPT {
    delete this;
}

const char* ActivationLogisticRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void ActivationLogisticRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

bool ActivationLogisticRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
    //todo assert
}

IPluginV2* ActivationLogisticRT::clone() const NOEXCEPT {
    auto *p = new ActivationLogisticRT();
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

ActivationLogisticRTPluginCreator::ActivationLogisticRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ActivationLogisticRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2* ActivationLogisticRTPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                                size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ActivationLogisticRT(serialData, serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char* ActivationLogisticRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2* ActivationLogisticRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    auto *pluginObj = new ActivationLogisticRT();
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char* ActivationLogisticRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection* ActivationLogisticRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}

const char *ActivationLogisticRTPluginCreator::getPluginName() const NOEXCEPT  {
    return "ActivationLogisticRT_tkDNN";
}