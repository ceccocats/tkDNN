#include <tkDNN/pluginsRT/ReshapeRT.h>
using namespace nvinfer1;

std::vector<PluginField> ReshapeRTPluginCreator::mPluginAttributes;
PluginFieldCollection ReshapeRTPluginCreator::mFC{};

ReshapeRT::ReshapeRT(dataDim_t newDim) {
    new_dim = newDim;
    n = new_dim.n;
    c = new_dim.c;
    h = new_dim.h;
    w = new_dim.w;
}

ReshapeRT::ReshapeRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
    new_dim.n = readBUF<int>(buf);
    new_dim.c = readBUF<int>(buf);
    new_dim.h = readBUF<int>(buf);
    new_dim.w = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

ReshapeRT::~ReshapeRT() {}

int ReshapeRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ReshapeRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{ c,h,w} ;
}

void ReshapeRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {}

int ReshapeRT::initialize() NOEXCEPT {
    return 0;
}

void ReshapeRT::terminate() NOEXCEPT {}

size_t ReshapeRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int ReshapeRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                       cudaStream_t stream) NOEXCEPT {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

    checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t ReshapeRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    std::cout << new_dim.c << ":" << new_dim.h << std::endl;
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
    std::cout << "C : " << c << "H : " << h << "w :" << w << std::endl;
    checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*new_dim.c*new_dim.h*new_dim.w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
    return 0;
}
#endif


size_t ReshapeRT::getSerializationSize() const NOEXCEPT {
    return 4*sizeof(int);
}

void ReshapeRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a = buf;
    writeBUF(buf, n);
    writeBUF(buf, c);
    writeBUF(buf, h);
    writeBUF(buf, w);
    assert(buf == a + getSerializationSize());
}

bool ReshapeRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
    //todo assert
}

const char *ReshapeRT::getPluginType() const NOEXCEPT {
    return "ReshapeRT_tkDNN";
}

const char *ReshapeRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

void ReshapeRT::destroy() NOEXCEPT {
    delete this;
}

const char *ReshapeRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void ReshapeRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2 *ReshapeRT::clone() const NOEXCEPT {
    auto *p = new ReshapeRT(new_dim);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

ReshapeRTPluginCreator::ReshapeRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ReshapeRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ReshapeRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2 *ReshapeRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ReshapeRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *ReshapeRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    dataDim_t newDim = *(static_cast<const dataDim_t *>(fields[0].data));
    ReshapeRT *pluginObj = new ReshapeRT(newDim);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ReshapeRTPluginCreator::getPluginName() const NOEXCEPT {
    return "ReshapeRT_tkDNN";
}

const char *ReshapeRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *ReshapeRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}














