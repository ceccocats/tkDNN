#include <tkDNN/pluginsRT/ResizeLayerRT.h>
using namespace nvinfer1;

std::vector<PluginField> ResizeLayerRTPluginCreator::mPluginAttributes;
PluginFieldCollection ResizeLayerRTPluginCreator::mFC{};


ResizeLayerRT::ResizeLayerRT(int oc, int oh, int ow,int ic,int ih,int iw) {
   this->o_c = oc;
   this->o_h = oh;
   this->o_w = ow;
   this->i_c = ic;
   this->i_h = ih;
   this->i_w = iw;
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
#elif NV_TENSORRT_MAJOR <= 7
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
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
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

IPluginV2Ext *ResizeLayerRT::clone() const NOEXCEPT {
    auto *p = new ResizeLayerRT(o_c,o_h,o_w,i_c,i_h,i_w);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

DataType
ResizeLayerRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void ResizeLayerRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                    IGpuAllocator *gpuAllocator) NOEXCEPT {

}

bool ResizeLayerRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                 int nbInputs) const NOEXCEPT {
    return false;
}

bool ResizeLayerRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void ResizeLayerRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                                    const DataType *inputTypes, const DataType *outputTypes,
                                    const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                                    PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT {

}

void ResizeLayerRT::detachFromContext() NOEXCEPT {

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

IPluginV2Ext *ResizeLayerRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new ResizeLayerRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *ResizeLayerRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    assert(fc->nbFields == 6);
    for(int i=0;i<6;i++){
        assert(fields[i].type == PluginFieldType::kINT32);
    }
    int oc = *(static_cast<const int *>(fields[0].data));
    int oh = *(static_cast<const int *>(fields[1].data));
    int ow = *(static_cast<const int *>(fields[2].data));
    int ic = *(static_cast<const int *>(fields[3].data));
    int ih = *(static_cast<const int *>(fields[4].data));
    int iw = *(static_cast<const int *>(fields[5].data));
    auto *pluginObj = new ResizeLayerRT(oc,oh,ow,ic,ih,iw);
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












