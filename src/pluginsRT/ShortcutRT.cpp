#include <tkDNN/pluginsRT/ShortcutRT.h>
using namespace nvinfer1;

std::vector<PluginField> ShortcutRTPluginCreator::mPluginAttributes;
PluginFieldCollection ShortcutRTPluginCreator::mFC{};

ShortcutRT::ShortcutRT(tk::dnn::dataDim_t bdim, bool mul) {
    bDim = bdim;
    this->bc = bDim.c;
    this->bh = bDim.h;
    this->bw = bDim.w;
    this->mul = mul;
}

ShortcutRT::~ShortcutRT() {}

ShortcutRT::ShortcutRT(const void *data, size_t length) {
    const char* buf =reinterpret_cast<const char*>(data),*bufCheck = buf;
    bDim.c = readBUF<int>(buf);
    bDim.h = readBUF<int>(buf);
    bDim.w = readBUF<int>(buf);
    bDim.l = 1;
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

void ShortcutRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                                     DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {
    c = inputDims[0].d[0];
    h = inputDims[0].d[1];
    w = inputDims[0].d[2];
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
#elif NV_TENSORRT_MAJOR == 7
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
    return "ShortcutRT_tkDNN";
}

const char *ShortcutRT::getPluginVersion() const NOEXCEPT {
    return "1";
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

IPluginV2 *ShortcutRT::clone() const NOEXCEPT {
    auto *p = new ShortcutRT(bDim,mul);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

ShortcutRTPluginCreator::ShortcutRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ShortcutRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ShortcutRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2 *ShortcutRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ShortcutRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *ShortcutRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    //todo assert
    tk::dnn::dataDim_t bdim = *(static_cast<const tk::dnn::dataDim_t *>(fields[0].data));
    bool mul = *(static_cast<const bool *>(fields[1].data));
    auto *pluginObj = new ShortcutRT(bdim,mul);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ShortcutRTPluginCreator::getPluginName() const NOEXCEPT {
    return "ShortcutRT_tkDNN";
}

const char *ShortcutRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *ShortcutRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}














