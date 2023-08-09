//
// Created by Adam on 4/11/2022
//
#include <tkDNN/pluginsRT/ActivationSwishRT.h>
using namespace  nvinfer1;
std::vector<PluginField> ActivationSwishRTPluginCreator::mPluginAttributes;
PluginFieldCollection ActivationSwishRTPluginCreator::mFC{};

ActivationSwishRT::ActivationSwishRT() {

}

ActivationSwishRT::~ActivationSwishRT() {

}

ActivationSwishRT::ActivationSwishRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char *>(data), *bufCheck = buf;
    size = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

int ActivationSwishRT::getNbOutputs() const NOEXCEPT  { return 1; }

Dims ActivationSwishRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT  { return inputs[0]; }

void ActivationSwishRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                         PluginFormat format, int maxBatchSize) NOEXCEPT {
assert(format == PluginFormat::kLINEAR);
size = 1;
for (int i = 0; i < outputDims[0].nbDims; i++)
size *= outputDims[0].d[i];
}

int ActivationSwishRT::initialize() NOEXCEPT  { return 0; }

void ActivationSwishRT::terminate() NOEXCEPT {}

size_t ActivationSwishRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT  { return 0; }

#if NV_TENSORRT_MAJOR > 7
int ActivationSwishRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
            cudaStream_t stream) NOEXCEPT {
    activationSwishForward((dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
    reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, stream);
    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t ActivationSwishRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                                  cudaStream_t stream) {
    activationSwishForward((dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
                          reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, stream);
    return 0;
}
#endif

size_t ActivationSwishRT::getSerializationSize() const NOEXCEPT  {
    return 1 * sizeof(int);
}

void ActivationSwishRT::serialize(void *buffer) const NOEXCEPT  {
    char *buf = reinterpret_cast<char *>(buffer), *a = buf;
    writeBUF(buf, size);
    assert(buf == a + getSerializationSize());
}

const char* ActivationSwishRT::getPluginType() const NOEXCEPT  {
    return "ActivationSwishRT_tkDNN";
}

const char *ActivationSwishRT::getPluginVersion() const NOEXCEPT  {
    return "1";
}

bool ActivationSwishRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
        return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char *ActivationSwishRT::getPluginNamespace() const NOEXCEPT  {
    return mPluginNamespace.c_str();
}

void ActivationSwishRT::setPluginNamespace(const char *plguinNamespace) NOEXCEPT  {
    mPluginNamespace = plguinNamespace;
}

IPluginV2 *ActivationSwishRT::clone() const NOEXCEPT {
    auto *p = new ActivationSwishRT();
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}



ActivationSwishRTPluginCreator::ActivationSwishRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ActivationSwishRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ActivationSwishRTPluginCreator::getPluginNamespace() const NOEXCEPT  {
    return mPluginNamespace.c_str();
}

IPluginV2 *ActivationSwishRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT  {
    auto *pluginObj = new ActivationSwishRT(serialData, serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *ActivationSwishRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT  {
    const PluginField *fields = fc->fields;
    auto *pluginObj = new ActivationSwishRT();
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ActivationSwishRTPluginCreator::getPluginName() const NOEXCEPT  {
    return "ActivationSwishRT_tkDNN";
}

const char *ActivationSwishRTPluginCreator::getPluginVersion() const NOEXCEPT{
    return "1";
}

const PluginFieldCollection *ActivationSwishRTPluginCreator::getFieldNames() NOEXCEPT  {
    return &mFC;
}

