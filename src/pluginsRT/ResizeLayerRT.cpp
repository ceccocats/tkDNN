#include <tkDNN/pluginsRT/ResizeLayerRT.h>
using namespace nvinfer1;

std::vector<PluginField> ResizeLayerRTPluginCreator::mPluginAttributes;
PluginFieldCollection ResizeLayerRTPluginCreator::mFC{};


ResizeLayerRT::ResizeLayerRT(int c, int h, int w) {
    o_c = c;
    o_h = h;
    o_w = w;
}

ResizeLayerRT::ResizeLayerRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
    o_c = readBUF<int>(buf);
    o_h = readBUF<int>(buf);
    o_w = readBUF<int>(buf);
    i_c = readBUF<int>(buf);
    i_h = readBUF<int>(buf);
    i_w = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

ResizeLayerRT::~ResizeLayerRT() {}

int ResizeLayerRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ResizeLayerRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{o_c, o_h, o_w};
}

void ResizeLayerRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                        DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {
    i_c = inputDims[0].d[0];
    i_h = inputDims[0].d[1];
    i_w = inputDims[0].d[2];
}

int ResizeLayerRT::initialize() NOEXCEPT {
    return 0;
}

void ResizeLayerRT::terminate() NOEXCEPT {}

size_t ResizeLayerRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT { return 0; }

#if NV_TENSORRT_MAJOR > 7
int ResizeLayerRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                           cudaStream_t stream) NOEXCEPT {
    resizeForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
                  reinterpret_cast<dnnType*>(outputs[0]),
                  batchSize, i_c, i_h, i_w, o_c, o_h, o_w, stream);
    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t ResizeLayerRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                               cudaStream_t stream) {
    resizeForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
                  reinterpret_cast<dnnType*>(outputs[0]),
                  batchSize, i_c, i_h, i_w, o_c, o_h, o_w, stream);
    return 0;
}
#endif

size_t ResizeLayerRT::getSerializationSize() const NOEXCEPT {
    return 6*sizeof(int);
}

void ResizeLayerRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, o_c);
    writeBUF(buf, o_h);
    writeBUF(buf, o_w);
    writeBUF(buf, i_c);
    writeBUF(buf, i_h);
    writeBUF(buf, i_w);
    assert(buf == a + getSerializationSize());
}

bool ResizeLayerRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
    //todo assert
}

const char *ResizeLayerRT::getPluginType() const NOEXCEPT {
    return "ResizeLayerRT_tkDNN";
}

const char *ResizeLayerRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

void ResizeLayerRT::destroy() NOEXCEPT {
    delete this;
}

const char *ResizeLayerRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void ResizeLayerRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2 *ResizeLayerRT::clone() const NOEXCEPT {
    auto *p = new ResizeLayerRT(o_c,o_h,o_w);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

ResizeLayerRTPluginCreator::ResizeLayerRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ResizeLayerRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ResizeLayerRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2 *ResizeLayerRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ResizeLayerRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *ResizeLayerRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    assert(fc->nbFields == 3);
    assert(fields[0].type == PluginFieldType::kINT32);
    assert(fields[1].type == PluginFieldType::kINT32);
    assert(fields[2].type == PluginFieldType::kINT32);
    int oc = *(static_cast<const int *>(fields[0].data));
    int oh = *(static_cast<const int *>(fields[1].data));
    int ow = *(static_cast<const int *>(fields[2].data));
    auto *pluginObj = new ResizeLayerRT(oc,oh,ow);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ResizeLayerRTPluginCreator::getPluginName() const NOEXCEPT {
    return "ResizeLayerRT_tkDNN";
}

const char *ResizeLayerRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *ResizeLayerRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}












