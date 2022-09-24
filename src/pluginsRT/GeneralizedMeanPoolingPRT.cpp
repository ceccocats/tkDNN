#include <tkDNN/pluginsRT/GeneralizedMeanPoolingPRT.h>
using namespace nvinfer1;

std::vector<PluginField> GeneralizedMeanPoolingPRTPluginCreator::mPluginAttributes;
PluginFieldCollection GeneralizedMeanPoolingPRTPluginCreator::mFC{};

static const char* GENERALIZEDMEANPOOLINGPRT_PLUGIN_VERSION{"1"};
static const char* GENERALIZEDMEANPOOLINGPRT_PLUGIN_NAME{"GeneralizedMeanPoolingPRT_tkDNN"};

GeneralizedMeanPoolingPRT::GeneralizedMeanPoolingPRT(int input_c, int input_h, int input_w, int input_n, int output_c, int output_h, int output_w, int output_n, float p){
    this->i_c = input_c;
    this->i_h = input_h;
    this->i_w = input_w;
    this->i_n = input_n;
    this->o_c = output_c;
    this->o_h = output_h;
    this->o_w = output_w;
    this->o_n = output_n;
    this->p = p;
}

GeneralizedMeanPoolingPRT::GeneralizedMeanPoolingPRT(const void *data, size_t length) {
    const char *buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
    i_c = readBUF<int>(buf);
    i_h = readBUF<int>(buf);
    i_w = readBUF<int>(buf);
    i_n = readBUF<int>(buf);
    o_c = readBUF<int>(buf);
    o_h = readBUF<int>(buf);
    o_w = readBUF<int>(buf);
    o_n = readBUF<int>(buf);
    p = readBUF<float>(buf);
    assert(buf == bufCheck + length);
}

GeneralizedMeanPoolingPRT::~GeneralizedMeanPoolingPRT() {

}

int GeneralizedMeanPoolingPRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims GeneralizedMeanPoolingPRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{this->o_c, this->o_h, this->o_w};
}

int GeneralizedMeanPoolingPRT::initialize() NOEXCEPT {
    return 0;
}

void GeneralizedMeanPoolingPRT::terminate() NOEXCEPT {

}

size_t GeneralizedMeanPoolingPRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int GeneralizedMeanPoolingPRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                                cudaStream_t stream) NOEXCEPT {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
    GeneralizedMeanPoolingP(srcData, dstData, batchSize, this->i_c, this->i_h, this->i_w, this->p, stream);
    return 0;
}
#elif NV_TENSORRT_MAJOR <= 7
int32_t GeneralizedMeanPoolingPRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                                    cudaStream_t stream) {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
    GeneralizedMeanPoolingP(srcData, dstData, batchSize, this->i_c, this->i_h, this->i_w, this->p, stream);
    return 0;
}
#endif


size_t GeneralizedMeanPoolingPRT::getSerializationSize() const NOEXCEPT {
    return 8*sizeof(int);
}

void GeneralizedMeanPoolingPRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, this->i_c);
    writeBUF(buf, this->i_h);
    writeBUF(buf, this->i_w);
    writeBUF(buf, this->i_n);
    writeBUF(buf, this->o_c);
    writeBUF(buf, this->o_h);
    writeBUF(buf, this->o_w);
    writeBUF(buf, this->o_n);
    writeBUF(buf, this->p);
    assert(buf == a + getSerializationSize());
}

void GeneralizedMeanPoolingPRT::destroy() NOEXCEPT {
delete this;
}

bool GeneralizedMeanPoolingPRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

const char *GeneralizedMeanPoolingPRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void GeneralizedMeanPoolingPRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *GeneralizedMeanPoolingPRT::getPluginType() const NOEXCEPT {
    return GENERALIZEDMEANPOOLINGPRT_PLUGIN_NAME;
}

const char *GeneralizedMeanPoolingPRT::getPluginVersion() const NOEXCEPT {
    return GENERALIZEDMEANPOOLINGPRT_PLUGIN_VERSION;
}

IPluginV2Ext *GeneralizedMeanPoolingPRT::clone() const NOEXCEPT {
    auto *pl = new GeneralizedMeanPoolingPRT(i_c,i_h,i_w,i_n, o_c,o_h,o_w,o_n,p);
    pl->setPluginNamespace(mPluginNamespace.c_str());
    return pl;
}

DataType
GeneralizedMeanPoolingPRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void GeneralizedMeanPoolingPRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                         IGpuAllocator *gpuAllocator) NOEXCEPT {

}

bool GeneralizedMeanPoolingPRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                      int nbInputs) const NOEXCEPT {
    return false;
}

bool GeneralizedMeanPoolingPRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void
GeneralizedMeanPoolingPRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                                    const DataType *inputTypes, const DataType *outputTypes,
                                    const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                                    PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT {

}

void GeneralizedMeanPoolingPRT::detachFromContext() NOEXCEPT {
    IPluginV2Ext::detachFromContext();
}

GeneralizedMeanPoolingPRTPluginCreator::GeneralizedMeanPoolingPRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void GeneralizedMeanPoolingPRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *GeneralizedMeanPoolingPRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *GeneralizedMeanPoolingPRTPluginCreator::deserializePlugin(const char *name, const void *serialData,size_t serialLength) NOEXCEPT {
    auto *pluginObj = new GeneralizedMeanPoolingPRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *GeneralizedMeanPoolingPRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    int i_c = *(static_cast<const int *>(fields[0].data));
    int i_h = *(static_cast<const int *>(fields[1].data));
    int i_w = *(static_cast<const int *>(fields[2].data));
    int i_n = *(static_cast<const int *>(fields[3].data));
    int o_c = *(static_cast<const int *>(fields[4].data));
    int o_h = *(static_cast<const int *>(fields[5].data));
    int o_w = *(static_cast<const int *>(fields[6].data));
    int o_n = *(static_cast<const int *>(fields[7].data));
    float p = *(static_cast<const float *>(fields[8].data));
    auto *pluginObj = new GeneralizedMeanPoolingPRT(i_c,i_h,i_w,i_n, o_c,o_h,o_w,o_n,p);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *GeneralizedMeanPoolingPRTPluginCreator::getPluginName() const NOEXCEPT {
    return GENERALIZEDMEANPOOLINGPRT_PLUGIN_NAME;
}

const char *GeneralizedMeanPoolingPRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return GENERALIZEDMEANPOOLINGPRT_PLUGIN_VERSION;
}

const PluginFieldCollection *GeneralizedMeanPoolingPRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}












