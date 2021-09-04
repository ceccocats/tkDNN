#include <tkDNN/pluginsRT/ActivationReLUCeilingRT.h>
using namespace nvinfer1;

std::vector<PluginField> ActivationReLUCeilingPluginCreator::mPluginAttributes;
PluginFieldCollection ActivationReLUCeilingPluginCreator::mFC{};

ActivationReLUCeiling::ActivationReLUCeiling(const float ceiling) {
    this->ceiling = ceiling;
}

ActivationReLUCeiling::~ActivationReLUCeiling() {

}

ActivationReLUCeiling::ActivationReLUCeiling(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char *>(data), *bufCheck = buf;
    ceiling = readBUF<float>(buf);
    size = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

int ActivationReLUCeiling::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ActivationReLUCeiling::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT { return inputs[0]; }

void ActivationReLUCeiling::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {
    assert(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    size = 1;
    for (int i = 0; i < outputDims[0].nbDims; i++)
        size *= outputDims[0].d[i];
}

int ActivationReLUCeiling::initialize() NOEXCEPT  { return 0; }

void ActivationReLUCeiling::terminate() NOEXCEPT {}

size_t ActivationReLUCeiling::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

int ActivationReLUCeiling::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,cudaStream_t stream) NOEXCEPT  {
    activationReLUCeilingForward((dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
    reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, ceiling, stream);
    return 0;
}

size_t ActivationReLUCeiling::getSerializationSize() const NOEXCEPT {
    return 1 * sizeof(int) + 1 * sizeof(float);
}

void ActivationReLUCeiling::serialize(void *buffer) const NOEXCEPT  {
    char *buf = reinterpret_cast<char *>(buffer), *a = buf;
    writeBUF(buf, ceiling);
    writeBUF(buf, size);
    assert(buf = a + getSerializationSize());
}

IPluginV2 *ActivationReLUCeiling::clone() const NOEXCEPT {
    auto *p = new ActivationReLUCeiling(ceiling);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

bool ActivationReLUCeiling::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT  {
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

void ActivationReLUCeiling::destroy() NOEXCEPT { delete this; }

const char *ActivationReLUCeiling::getPluginType() const NOEXCEPT {
    return "ActivationReLUCeilingRT_tkDNN";
}

const char *ActivationReLUCeiling::getPluginVersion() const NOEXCEPT {
    return "1";
}

const char *ActivationReLUCeiling::getPluginNamespace() const NOEXCEPT  {
    return mPluginNamespace.c_str();
}

void ActivationReLUCeiling::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

ActivationReLUCeilingPluginCreator::ActivationReLUCeilingPluginCreator() {
    mPluginAttributes.emplace_back(PluginField("ceiling", nullptr, PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ActivationReLUCeilingPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT  {
    mPluginNamespace = pluginNamespace;
}

const char *ActivationReLUCeilingPluginCreator::getPluginNamespace() const NOEXCEPT  {
    return mPluginNamespace.c_str();
}

IPluginV2 *ActivationReLUCeilingPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ActivationReLUCeiling(serialData, serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *ActivationReLUCeilingPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT  {
    const PluginField *fields = fc->fields;
    float ceiling = *(static_cast<const float *>(fields[0].data));
    auto *pluginObj = new ActivationReLUCeiling(ceiling);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ActivationReLUCeilingPluginCreator::getPluginName() const NOEXCEPT  {
    return "ActivationReLUCeilingRT_tkDNN";
}

const char *ActivationReLUCeilingPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *ActivationReLUCeilingPluginCreator::getFieldNames() NOEXCEPT  {
    return &mFC;
}



