#include <tkDNN/pluginsRT/ReshapeRT.h>
using namespace nvinfer1;

std::vector<PluginField> ReshapeRTPluginCreator::mPluginAttributes;
PluginFieldCollection ReshapeRTPluginCreator::mFC{};

static const char* RESHAPERT_PLUGIN_VERSION{"1"};
static const char* RESHAPERT_PLUGIN_NAME{"ReshapeRT_tkDNN"};

ReshapeRT::ReshapeRT(int n,int c,int h,int w) {
    this->n = n;
    this->c = c;
    this->h = h;
    this->w = w;
}

ReshapeRT::ReshapeRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
    n = readBUF<int>(buf);
    c = readBUF<int>(buf);
    h = readBUF<int>(buf);
    w = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

ReshapeRT::~ReshapeRT() {}

int ReshapeRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims ReshapeRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{ c,h,w} ;
}

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
#elif NV_TENSORRT_MAJOR <= 7
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
    return RESHAPERT_PLUGIN_NAME;
}

const char *ReshapeRT::getPluginVersion() const NOEXCEPT {
    return RESHAPERT_PLUGIN_VERSION;
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

IPluginV2Ext *ReshapeRT::clone() const NOEXCEPT {
    auto *p = new ReshapeRT(n,c,h,w);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

DataType ReshapeRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void ReshapeRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                IGpuAllocator *gpuAllocator) NOEXCEPT {

}

bool
ReshapeRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const NOEXCEPT {
    return false;
}

bool ReshapeRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void ReshapeRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                                const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                const bool *outputIsBroadcast, PluginFormat floatFormat,
                                int32_t maxBatchSize) NOEXCEPT {

}

void ReshapeRT::detachFromContext() NOEXCEPT {

}

ReshapeRTPluginCreator::ReshapeRTPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("n", nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("c", nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("h", nullptr,PluginFieldType::kINT32,1));
    mPluginAttributes.emplace_back(PluginField("w", nullptr,PluginFieldType::kINT32,1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void ReshapeRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *ReshapeRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *ReshapeRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ReshapeRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *ReshapeRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    assert(fc->nbFields == 4);
    for(int i=0;i<4;i++){
        assert(fields[1].type == PluginFieldType::kINT32);
    }
    int n = *(static_cast<const int *>(fields[0].data));
    int c = *(static_cast<const int *>(fields[1].data));
    int h = *(static_cast<const int *>(fields[2].data));
    int w = *(static_cast<const int *>(fields[3].data));

    auto *pluginObj = new ReshapeRT(n,c,h,w);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *ReshapeRTPluginCreator::getPluginName() const NOEXCEPT {
    return RESHAPERT_PLUGIN_NAME;
}

const char *ReshapeRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return RESHAPERT_PLUGIN_VERSION;
}

const PluginFieldCollection *ReshapeRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}














