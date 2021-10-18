#include <tkDNN/pluginsRT/UpsampleRT.h>
using namespace nvinfer1;

std::vector<PluginField> UpsampleRTPluginCreator::mPluginAttributes;
PluginFieldCollection UpsampleRTPluginCreator::mFC{};

UpsampleRT::UpsampleRT(int stride) {
    this->stride = stride;
}

UpsampleRT::UpsampleRT(const void *data, size_t length) {
    const char* buf = reinterpret_cast<const char*>(data),*bufCheck=buf;
    stride = readBUF<int>(buf);
    c = readBUF<int>(buf);
    h = readBUF<int>(buf);
    w = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

UpsampleRT::~UpsampleRT() {}

int UpsampleRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims UpsampleRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3(inputs[0].d[0], inputs[0].d[1]*stride, inputs[0].d[2]*stride);
}

void UpsampleRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                     DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {
    c = inputDims[0].d[0];
    h = inputDims[0].d[1];
    w = inputDims[0].d[2];
}

int UpsampleRT::initialize() NOEXCEPT {
    return 0;
}

void UpsampleRT::terminate() NOEXCEPT {}

size_t UpsampleRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int UpsampleRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) NOEXCEPT {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

    fill(dstData, batchSize*c*h*w*stride*stride, 0.0, stream);
    upsampleForward(srcData, dstData, batchSize, c, h, w, stride, 1, 1, stream);
    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t UpsampleRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                            cudaStream_t stream) {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

    fill(dstData, batchSize*c*h*w*stride*stride, 0.0, stream);
    upsampleForward(srcData, dstData, batchSize, c, h, w, stride, 1, 1, stream);
    return 0;
}
#endif

size_t UpsampleRT::getSerializationSize() const NOEXCEPT {
    return 4*sizeof(int);
}

void UpsampleRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, stride);
    writeBUF(buf, c);
    writeBUF(buf, h);
    writeBUF(buf, w);
    assert(buf == a + getSerializationSize());
}

bool UpsampleRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
}

const char *UpsampleRT::getPluginType() const NOEXCEPT {
    return "Upsample_tkDNN";
}

const char *UpsampleRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

void UpsampleRT::destroy() NOEXCEPT {
    delete this;
}

const char *UpsampleRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void UpsampleRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2 *UpsampleRT::clone() const NOEXCEPT {
    auto *p = new UpsampleRT(stride);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

UpsampleRTPluginCreator::UpsampleRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void UpsampleRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *UpsampleRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2 *UpsampleRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new UpsampleRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *UpsampleRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    int stride = *(static_cast<const int *>(fields[0].data));
    auto *pluginObj = new UpsampleRT(stride);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *UpsampleRTPluginCreator::getPluginName() const NOEXCEPT {
    return "Upsample_tkDNN";
}

const char *UpsampleRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *UpsampleRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}
















