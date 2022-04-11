#include <tkDNN/pluginsRT/ActivationLeakyRT.h>
using namespace nvinfer1;


std::vector<PluginField> ActivationLeakyRTPluginCreator::mPluginAttributes;
PluginFieldCollection ActivationLeakyRTPluginCreator::mFC{};

ActivationLeakyRT::ActivationLeakyRT(float s) {
    slope = s;
}

ActivationLeakyRT::ActivationLeakyRT(const void *data, size_t length) {
    std::cout << "DESERIALIZE LEAKYRT" << std::endl;
    const char *buf = reinterpret_cast<const char *>(data), *bufCheck = buf;
    slope = readBUF<float>(buf);
    size = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

ActivationLeakyRT::~ActivationLeakyRT() {}

int ActivationLeakyRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ActivationLeakyRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return inputs[0];
}

void ActivationLeakyRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                            DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {
    assert(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    size = 1;
    for (int i = 0; i < outputDims[0].nbDims; i++)
        size *= outputDims[0].d[i];
}
int ActivationLeakyRT::initialize() NOEXCEPT {
    return 0;
}

size_t ActivationLeakyRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int ActivationLeakyRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                               cudaStream_t stream) NOEXCEPT {
    activationLEAKYForward(
            (dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
            reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, slope,
            stream);
    return 0;

}
#elif NV_TENSORRT_MAJOR == 7
int32_t ActivationLeakyRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                                   cudaStream_t stream) {
    activationLEAKYForward(
            (dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
            reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, slope,
            stream);
    return 0;
}
#endif


size_t ActivationLeakyRT::getSerializationSize() const NOEXCEPT {
    return 1 * sizeof(int) + 1 * sizeof(float);
}

void ActivationLeakyRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char *>(buffer), *a = buf;
    writeBUF(buf, size);
    assert(buf == a + getSerializationSize());
}

bool ActivationLeakyRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char *ActivationLeakyRT::getPluginType() const NOEXCEPT {
    return "ActivationLeakyRT_tkDNN";
}

const char *ActivationLeakyRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

void ActivationLeakyRT::destroy() NOEXCEPT {
    delete this;
}

const char *ActivationLeakyRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();

}

void ActivationLeakyRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2* ActivationLeakyRT::clone() const NOEXCEPT {
    auto *p = new ActivationLeakyRT(slope);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

ActivationLeakyRTPluginCreator::ActivationLeakyRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ActivationLeakyRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2* ActivationLeakyRTPluginCreator::deserializePlugin(const char *name, const void *serialData,size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ActivationLeakyRT(serialData, serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char* ActivationLeakyRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2* ActivationLeakyRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    assert(fc->nbFields == 1);
    assert(fields[0].type == PluginFieldType::kFLOAT32);
    float slope = *(static_cast<const float *>(fields[0].data));
    auto *pluginObj = new ActivationLeakyRT(slope);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char* ActivationLeakyRTPluginCreator::getPluginName() const NOEXCEPT {
    return "ActivationLeakyRT_tkDNN";
}

const char* ActivationLeakyRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection* ActivationLeakyRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}

