//
// Created by perseusdg on 9/4/21.
//
#include <tkDNN/pluginsRT/ActivationMishRT.h>
using namespace  nvinfer1;
std::vector<PluginField> ActivationMishRTPluginCreator::mPluginAttributes;
PluginFieldCollection ActivationMishRTPluginCreator::mFC{};

ActivationMishRT::ActivationMishRT() {

}

ActivationMishRT::~ActivationMishRT() {

}

ActivationMishRT::ActivationMishRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char *>(data), *bufCheck = buf;
    size = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

int ActivationMishRT::getNbOutputs() const NOEXCEPT  { return 1; }

Dims ActivationMishRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT  { return inputs[0]; }

void ActivationMishRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                         PluginFormat format, int maxBatchSize) NOEXCEPT {
assert(format == PluginFormat::kLINEAR);
size = 1;
for (int i = 0; i < outputDims[0].nbDims; i++)
size *= outputDims[0].d[i];
}

int ActivationMishRT::initialize() NOEXCEPT  { return 0; }

void ActivationMishRT::terminate() NOEXCEPT {}

size_t ActivationMishRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT  { return 0; }

#if NV_TENSORRT_MAJOR > 7
int ActivationMishRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
            cudaStream_t stream) NOEXCEPT {
    activationMishForward((dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
    reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, stream);
    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t ActivationMishRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                                  cudaStream_t stream) {
    activationMishForward((dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
                          reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, stream);
    return 0;
}
#endif

size_t ActivationMishRT::getSerializationSize() const NOEXCEPT  {
    return 1 * sizeof(int);
}

void ActivationMishRT::serialize(void *buffer) const NOEXCEPT  {
    char *buf = reinterpret_cast<char *>(buffer), *a = buf;
    writeBUF(buf, size);
    assert(buf == a + getSerializationSize());
}

const char* ActivationMishRT::getPluginType() const NOEXCEPT  {
    return "ActivationMishRT_tkDNN";
}

const char *ActivationMishRT::getPluginVersion() const NOEXCEPT  {
    return "1";
}

bool ActivationMishRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
        return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char *ActivationMishRT::getPluginNamespace() const NOEXCEPT  {
    return mPluginNamespace.c_str();
}

void ActivationMishRT::setPluginNamespace(const char *plguinNamespace) NOEXCEPT  {
    mPluginNamespace = plguinNamespace;
}

IPluginV2 *ActivationMishRT::clone() const NOEXCEPT {
    auto *p = new ActivationMishRT();
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}



ActivationMishRTPluginCreator::ActivationMishRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ActivationMishRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ActivationMishRTPluginCreator::getPluginNamespace() const NOEXCEPT  {
    return mPluginNamespace.c_str();
}

IPluginV2 *ActivationMishRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT  {
    auto *pluginObj = new ActivationMishRT(serialData, serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *ActivationMishRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT  {
    const PluginField *fields = fc->fields;
    auto *pluginObj = new ActivationMishRT();
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ActivationMishRTPluginCreator::getPluginName() const NOEXCEPT  {
    return "ActivationMishRT_tkDNN";
}

const char *ActivationMishRTPluginCreator::getPluginVersion() const NOEXCEPT{
    return "1";
}

const PluginFieldCollection *ActivationMishRTPluginCreator::getFieldNames() NOEXCEPT  {
    return &mFC;
}

