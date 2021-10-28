#include <tkDNN/pluginsRT/MaxPoolingFixedSizeRT.h>
using namespace nvinfer1;

std::vector<PluginField> MaxPoolFixedSizeRTPluginCreator::mPluginAttributes;
PluginFieldCollection MaxPoolFixedSizeRTPluginCreator::mFC{};

MaxPoolFixedSizeRT::MaxPoolFixedSizeRT(int c, int h, int w, int n, int strideH, int strideW, int winSize, int padding){
    this->c = c;
    this->h = h;
    this->w = w;
    this->n = n;
    this->stride_H = strideH;
    this->stride_W = strideW;
    this->winSize = winSize;
    this->padding = padding;
}

MaxPoolFixedSizeRT::MaxPoolFixedSizeRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
    c = readBUF<int>(buf);
    h = readBUF<int>(buf);
    w = readBUF<int>(buf);
    n = readBUF<int>(buf);
    stride_H = readBUF<int>(buf);
    stride_W = readBUF<int>(buf);
    winSize  = readBUF<int>(buf);
    padding = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

MaxPoolFixedSizeRT::~MaxPoolFixedSizeRT() {

}

int MaxPoolFixedSizeRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims MaxPoolFixedSizeRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{this->c, this->h, this->w};
}

int MaxPoolFixedSizeRT::initialize() NOEXCEPT {
    return 0;
}

void MaxPoolFixedSizeRT::terminate() NOEXCEPT {

}

size_t MaxPoolFixedSizeRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int MaxPoolFixedSizeRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                                cudaStream_t stream) NOEXCEPT {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
    MaxPoolingForward(srcData, dstData, batchSize, this->c, this->h, this->w, this->stride_H, this->stride_W, this->winSize, this->padding, stream);
    return 0;
}
#elif NV_TENSORRT_MAJOR <= 7
int32_t MaxPoolFixedSizeRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                                    cudaStream_t stream) {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
    MaxPoolingForward(srcData, dstData, batchSize, this->c, this->h, this->w, this->stride_H, this->stride_W, this->winSize, this->padding, stream);
    return 0;
}
#endif


size_t MaxPoolFixedSizeRT::getSerializationSize() const NOEXCEPT {
    return 8*sizeof(int);
}

void MaxPoolFixedSizeRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, this->c);
    writeBUF(buf, this->h);
    writeBUF(buf, this->w);
    writeBUF(buf, this->n);
    writeBUF(buf, this->stride_H);
    writeBUF(buf, this->stride_W);
    writeBUF(buf, this->winSize);
    writeBUF(buf, this->padding);
    assert(buf == a + getSerializationSize());
}

void MaxPoolFixedSizeRT::destroy() NOEXCEPT {
delete this;
}

bool MaxPoolFixedSizeRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
    //todo assert
}

const char *MaxPoolFixedSizeRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void MaxPoolFixedSizeRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *MaxPoolFixedSizeRT::getPluginType() const NOEXCEPT {
    return "MaxPoolingFixedSizeRT_tkDNN";
}

const char *MaxPoolFixedSizeRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

IPluginV2Ext *MaxPoolFixedSizeRT::clone() const NOEXCEPT {
    auto *p = new MaxPoolFixedSizeRT(c,h,w,n,stride_H,stride_W,winSize,padding);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

DataType
MaxPoolFixedSizeRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void MaxPoolFixedSizeRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                         IGpuAllocator *gpuAllocator) NOEXCEPT {

}

bool MaxPoolFixedSizeRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                      int nbInputs) const NOEXCEPT {
    return false;
}

bool MaxPoolFixedSizeRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void
MaxPoolFixedSizeRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                                    const DataType *inputTypes, const DataType *outputTypes,
                                    const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                                    PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT {

}

void MaxPoolFixedSizeRT::detachFromContext() NOEXCEPT {
    IPluginV2Ext::detachFromContext();
}

MaxPoolFixedSizeRTPluginCreator::MaxPoolFixedSizeRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void MaxPoolFixedSizeRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *MaxPoolFixedSizeRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *MaxPoolFixedSizeRTPluginCreator::deserializePlugin(const char *name, const void *serialData,size_t serialLength) NOEXCEPT {
    auto *pluginObj = new MaxPoolFixedSizeRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *MaxPoolFixedSizeRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    int c = *(static_cast<const int *>(fields[0].data));
    int h = *(static_cast<const int *>(fields[1].data));
    int w = *(static_cast<const int *>(fields[2].data));
    int n = *(static_cast<const int *>(fields[3].data));
    int stride_H = *(static_cast<const int *>(fields[4].data));
    int stride_W = *(static_cast<const int *>(fields[5].data));
    int winSize = *(static_cast<const int *>(fields[6].data));
    int padding = *(static_cast<const int *>(fields[7].data));
    auto *pluginObj = new MaxPoolFixedSizeRT(c,h,w,n,stride_H,stride_W,winSize,padding);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *MaxPoolFixedSizeRTPluginCreator::getPluginName() const NOEXCEPT {
    return "MaxPoolingFixedSizeRT_tkDNN";
}

const char *MaxPoolFixedSizeRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *MaxPoolFixedSizeRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}












