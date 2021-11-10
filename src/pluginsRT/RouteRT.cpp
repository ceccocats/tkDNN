#include <tkDNN/pluginsRT/RouteRT.h>
using namespace nvinfer1;

std::vector<PluginField> RouteRTPluginCreator::mPluginAttributes;
PluginFieldCollection RouteRTPluginCreator::mFC{};


RouteRT::RouteRT(int groups, int group_id) {
    this->groups = groups;
    this->group_id = group_id;
}

RouteRT::~RouteRT() {}

RouteRT::RouteRT(const void *data, size_t length) {
    const char* buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
    groups = readBUF<int>(buf);
    group_id = readBUF<int>(buf);
    in = readBUF<int>(buf);
    for(int i=0;i <MAX_INPUTS;i++){
        c_in[i] = readBUF<int>(buf);
    }
    c= readBUF<int>(buf);
    h = readBUF<int>(buf);
    w = readBUF<int>(buf);
    assert(buf == bufCheck + length);
}

int RouteRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims RouteRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    int out_c = 0;
    for(int i=0; i<nbInputDims; i++) out_c += inputs[i].d[0];
    return Dims3{out_c/groups, inputs[0].d[1], inputs[0].d[2]};
}

void
RouteRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                             PluginFormat format, int maxBatchSize) NOEXCEPT {
    in = nbInputs;
    c = 0;
    for(int i=0; i<nbInputs; i++) {
        c_in[i] = inputDims[i].d[0];
        c += inputDims[i].d[0];
    }
    h = inputDims[0].d[1];
    w = inputDims[0].d[2];
    c /= groups;
}

int RouteRT::initialize() NOEXCEPT {
    return 0;
}

void RouteRT::terminate() NOEXCEPT {}

size_t RouteRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int RouteRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                     cudaStream_t stream) NOEXCEPT {
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
    for(int b=0; b<batchSize; b++) {
        int offset = 0;
        for(int i=0; i<in; i++) {
            dnnType *input = (dnnType*)reinterpret_cast<const dnnType*>(inputs[i]);
            int in_dim = c_in[i]*h*w;
            int part_in_dim = in_dim / this->groups;
            checkCuda( cudaMemcpyAsync(dstData + b*c*w*h + offset, input + b*c*w*h*groups + this->group_id*part_in_dim, part_in_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream) );
            offset += part_in_dim;
        }
    }
    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t RouteRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
    for(int b=0; b<batchSize; b++) {
        int offset = 0;
        for(int i=0; i<in; i++) {
            dnnType *input = (dnnType*)reinterpret_cast<const dnnType*>(inputs[i]);
            int in_dim = c_in[i]*h*w;
            int part_in_dim = in_dim / this->groups;
            checkCuda( cudaMemcpyAsync(dstData + b*c*w*h + offset, input + b*c*w*h*groups + this->group_id*part_in_dim, part_in_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream) );
            offset += part_in_dim;
        }
    }
    return 0;
}
#endif

size_t RouteRT::getSerializationSize() const NOEXCEPT {
    return (6+MAX_INPUTS)*sizeof(int);
}

void RouteRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, groups);
    writeBUF(buf, group_id);
    writeBUF(buf, in);
    for(int i=0; i<MAX_INPUTS; i++)
        writeBUF(buf, c_in[i]);
    writeBUF(buf, c);
    writeBUF(buf, h);
    writeBUF(buf, w);
    assert(buf == a + getSerializationSize());
}

const char *RouteRT::getPluginType() const NOEXCEPT {
    return "RouteRT_tkDNN";
}

const char *RouteRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

void RouteRT::destroy() NOEXCEPT {
    delete this;
}

const char *RouteRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void RouteRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

bool RouteRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

IPluginV2 *RouteRT::clone() const NOEXCEPT {
    auto *p = new RouteRT(groups,group_id);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

RouteRTPluginCreator::RouteRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void RouteRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *RouteRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2 *RouteRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new RouteRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *RouteRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    assert(fc->nbFields == 2);
    assert(fields[0].type == PluginFieldType::kINT32);
    assert(fields[1].type == PluginFieldType::kINT32);
    int groups = *(static_cast<const int *>(fields[0].data));
    int group_id = *(static_cast<const int *>(fields[1].data));
    RouteRT *pluginObj = new RouteRT(groups,group_id);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *RouteRTPluginCreator::getPluginName() const NOEXCEPT {
    return "RouteRT_tkDNN";
}

const char *RouteRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *RouteRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}












