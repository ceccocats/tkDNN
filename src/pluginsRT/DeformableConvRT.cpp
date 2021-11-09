#include <tkDNN/pluginsRT/DeformableConvRT.h>

#include <utility>
using namespace nvinfer1;
using namespace tk::dnn;

std::vector<PluginField> DeformableConvRTPluginCreator::mPluginAttributes;
PluginFieldCollection DeformableConvRTPluginCreator::mFC{};

static const char* DEFORMABLECONVRT_PLUGIN_VERSION{"1"};
static const char* DEFORMABLECONVRT_PLUGIN_NAME{"DeformableConvRT_tkDNN"};


DeformableConvRT::DeformableConvRT(int chunk_dim, int kh, int kw, int sh, int sw, int ph, int pw, int deformableGroup,
                                   int i_n, int i_c, int i_h, int i_w, int o_n, int o_c, int o_h, int o_w,std::vector<dnnType> data_H,std::vector<dnnType> bias2_H,
                                   std::vector<dnnType> ones_d1_h,std::vector<dnnType> ones_d2_h,std::vector<dnnType> offsetH,std::vector<dnnType> maskH,int height_ones,int width_ones,int dim_ones) {
    this->chunk_dim = chunk_dim;
    this->kh = kh;
    this->kw = kw;
    this->sh = sh;
    this->sw = sw;
    this->ph = ph;
    this->pw = pw;
    this->deformableGroup = deformableGroup;
    this->i_n = i_n;
    this->i_c = i_c;
    this->i_h = i_h;
    this->i_w = i_w;
    this->o_n = o_n;
    this->o_c = o_c;
    this->o_h = o_h;
    this->o_w = o_w;
    this->mask_v = std::move(maskH);
    this->offset_v = std::move(offsetH);
    this->ones_d2_v = std::move(ones_d2_h);
    this->ones_d1_v = std::move(ones_d1_h);
    this->data_d_v = std::move(data_H);
    this->bias2_d_v = std::move(bias2_H);
    this->height_ones = height_ones;
    this->width_ones = width_ones;
    this->dim_ones = dim_ones;

    checkCuda( cudaMalloc(&data_d, i_c * o_c * kh * kw * 1 * sizeof(dnnType)));
    checkCuda( cudaMalloc(&bias2_d, o_c*sizeof(dnnType)));
    checkCuda( cudaMalloc(&ones_d1, height_ones * width_ones * sizeof(dnnType)));
    checkCuda( cudaMalloc(&offset, 2*chunk_dim*sizeof(dnnType)));
    checkCuda( cudaMalloc(&mask, chunk_dim*sizeof(dnnType)));
    checkCuda( cudaMalloc(&ones_d2, dim_ones*sizeof(dnnType)));
    if(!data_d_v.empty() && !bias2_d_v.empty() && !ones_d1_v.empty() && !ones_d2_v.empty() && !mask_v.empty() && !offset_v.empty()) {
        checkCuda(cudaMemcpy(data_d, data_d_v.data(), sizeof(dnnType) * data_d_v.size(), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(bias2_d, bias2_d_v.data(), sizeof(dnnType) * bias2_d_v.size(), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(ones_d1, ones_d1_v.data(), sizeof(dnnType) * ones_d1_v.size(), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(offset, offset_v.data(), sizeof(dnnType) * offset_v.size(), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(mask, mask_v.data(), sizeof(dnnType) * mask_v.size(), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(ones_d2, ones_d2_v.data(), sizeof(dnnType) * ones_d2_v.size(), cudaMemcpyHostToDevice));
    }


}

DeformableConvRT::~DeformableConvRT() {
    checkCuda( cudaFree(data_d) );
    checkCuda( cudaFree(bias2_d) );
    checkCuda( cudaFree(ones_d1) );
    checkCuda( cudaFree(offset) );
    checkCuda( cudaFree(mask) );
    checkCuda( cudaFree(ones_d2) );
}

DeformableConvRT::DeformableConvRT(const void *data, size_t length) {
    const char* buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
    chunk_dim = readBUF<int>(buf);
    kh = readBUF<int>(buf);
    kw = readBUF<int>(buf);
    sh = readBUF<int>(buf);
    sw = readBUF<int>(buf);
    ph = readBUF<int>(buf);
    pw = readBUF<int>(buf);
    deformableGroup = readBUF<int>(buf);
    i_n = readBUF<int>(buf);
    i_c = readBUF<int>(buf);
    i_h = readBUF<int>(buf);
    i_w = readBUF<int>(buf);
    o_n = readBUF<int>(buf);
    o_c = readBUF<int>(buf);
    o_h = readBUF<int>(buf);
    o_w = readBUF<int>(buf);
    height_ones = readBUF<int>(buf);
    width_ones = readBUF<int>(buf);
    dim_ones = readBUF<int>(buf);
    offset_v.resize(chunk_dim*2);
    for(int i=0;i<chunk_dim*2;i++)
        offset_v[i] = readBUF<dnnType>(buf);
    mask_v.resize(chunk_dim);
    for(int i=0;i<chunk_dim;i++)
        mask_v[i] = readBUF<dnnType>(buf);
    data_d_v.resize(i_c*o_c*kh*kw*1);
    for(int i=0;i<(i_c*o_c*kh*kw*1);i++)
        data_d_v[i] = readBUF<dnnType>(buf);
    bias2_d_v.resize(o_c);
    for(int i=0; i < o_c; i++)
        bias2_d_v[i] = readBUF<dnnType>(buf);
    ones_d1_v.resize(height_ones*width_ones);
    for(int i=0; i<height_ones * width_ones; i++)
        ones_d1_v[i] = readBUF<dnnType>(buf);
    ones_d2_v.resize(dim_ones);
    for(int i=0; i<dim_ones; i++)
        ones_d2_v[i] = readBUF<dnnType>(buf);
    assert(buf == bufCheck + length);

}

int DeformableConvRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims DeformableConvRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{o_c, o_h, o_w};
}

int DeformableConvRT::initialize() NOEXCEPT {
    return 0;
}

void DeformableConvRT::terminate() NOEXCEPT {}

size_t DeformableConvRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {return 0;}

#if NV_TENSORRT_MAJOR > 7
int DeformableConvRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                              cudaStream_t stream) NOEXCEPT {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *output_conv = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);

    // split conv2d outputs into offset to mask
    for(int b=0; b<batchSize; b++) {
        checkCuda(cudaMemcpy(offset, output_conv + b * 3 * chunk_dim, 2*chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice));
        checkCuda(cudaMemcpy(mask, output_conv + b * 3 * chunk_dim + 2*chunk_dim, chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice));
        // kernel sigmoid
        activationSIGMOIDForward(mask, mask, chunk_dim);
        // deformable convolution
        dcnV2CudaForward(stat, handle,
                         srcData, data_d,
                         bias2_d, ones_d1,
                         offset, mask,
                         reinterpret_cast<dnnType*>(outputs[0]), ones_d2,
                         kh, kw,
                         sh, sw,
                         ph, pw,
                         1, 1,
                         deformableGroup, b,
                         i_n, i_c, i_h, i_w,
                         o_n, o_c, o_h, o_w,
                         chunk_dim);
    }
    return 0;
}
#elif NV_TENSORRT_MAJOR <= 7
int32_t DeformableConvRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace,
                                  cudaStream_t stream) {
    dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
    dnnType *output_conv = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);

    // split conv2d outputs into offset to mask
    for(int b=0; b<batchSize; b++) {
        checkCuda(cudaMemcpy(offset, output_conv + b * 3 * chunk_dim, 2*chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice));
        checkCuda(cudaMemcpy(mask, output_conv + b * 3 * chunk_dim + 2*chunk_dim, chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice));
        // kernel sigmoid
        activationSIGMOIDForward(mask, mask, chunk_dim);
        // deformable convolution
        dcnV2CudaForward(stat, handle,
                         srcData, data_d,
                         bias2_d, ones_d1,
                         offset, mask,
                         reinterpret_cast<dnnType*>(outputs[0]), ones_d2,
                         kh, kw,
                         sh, sw,
                         ph, pw,
                         1, 1,
                         deformableGroup, b,
                         i_n, i_c, i_h, i_w,
                         o_n, o_c, o_h, o_w,
                         chunk_dim);
    }
    return 0;
}
#endif

size_t DeformableConvRT::getSerializationSize() const NOEXCEPT {
    return 19 * sizeof(int) + chunk_dim * 3 * sizeof(dnnType) + (i_c * o_c * kh * kw * 1 ) * sizeof(dnnType) +
           o_c * sizeof(dnnType) + height_ones * width_ones * sizeof(dnnType) + dim_ones * sizeof(dnnType);
}

void DeformableConvRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char*>(buffer),*a=buf;
    writeBUF(buf, chunk_dim);
    writeBUF(buf, kh);
    writeBUF(buf, kw);
    writeBUF(buf, sh);
    writeBUF(buf, sw);
    writeBUF(buf, ph);
    writeBUF(buf, pw);
    writeBUF(buf, deformableGroup);
    writeBUF(buf, i_n);
    writeBUF(buf, i_c);
    writeBUF(buf, i_h);
    writeBUF(buf, i_w);
    writeBUF(buf, o_n);
    writeBUF(buf, o_c);
    writeBUF(buf, o_h);
    writeBUF(buf, o_w);
    writeBUF(buf,height_ones);
    writeBUF(buf,width_ones);
    writeBUF(buf,dim_ones);
    for(int i=0; i<offset_v.size(); i++)
        writeBUF(buf, offset_v[i]);
    for(int i=0; i<mask_v.size(); i++)
        writeBUF(buf, mask_v[i]);
    for(int i=0; i<data_d_v.size(); i++)
        writeBUF(buf, data_d_v[i]);
    for(int i=0; i < bias2_d_v.size(); i++)
        writeBUF(buf, bias2_d_v[i]);
    for(int i=0; i<ones_d1_v.size(); i++)
        writeBUF(buf, ones_d1_v[i]);
    for(int i=0; i<ones_d2_v.size(); i++)
        writeBUF(buf, ones_d2_v[i]);

    assert(buf == a + getSerializationSize());
}

void DeformableConvRT::destroy() NOEXCEPT {
    delete this;
}


const char *DeformableConvRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void DeformableConvRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *DeformableConvRT::getPluginType() const NOEXCEPT {
    return DEFORMABLECONVRT_PLUGIN_NAME;
}

const char *DeformableConvRT::getPluginVersion() const NOEXCEPT {
    return DEFORMABLECONVRT_PLUGIN_VERSION;
}

IPluginV2Ext *DeformableConvRT::clone() const NOEXCEPT {
    auto *p = new DeformableConvRT(chunk_dim,kh,kw,sh,sw,ph,pw,deformableGroup,i_n,i_c,i_h,i_w,o_n,o_c,o_h,o_w,data_d_v,bias2_d_v,ones_d1_v,ones_d2_v,offset_v,mask_v,height_ones,width_ones,dim_ones);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

DataType
DeformableConvRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void DeformableConvRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                       IGpuAllocator *gpuAllocator) NOEXCEPT {
    handle = cublasContext;

}

bool DeformableConvRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                    int nbInputs) const NOEXCEPT {
    return false;
}

bool DeformableConvRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void DeformableConvRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                                  const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                                  const bool *outputIsBroadcast, PluginFormat floatFormat,
                                  int32_t maxBatchSize) NOEXCEPT {

}

void DeformableConvRT::detachFromContext() NOEXCEPT {

}

bool DeformableConvRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
}

DeformableConvRTPluginCreator::DeformableConvRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void DeformableConvRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *DeformableConvRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *DeformableConvRTPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                            size_t serialLength) NOEXCEPT {
    auto *pluginObj = new DeformableConvRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *DeformableConvRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    int chunk_dim = *(static_cast<const int *>(fields[0].data));
    int kh = *(static_cast<const int *>(fields[1].data));
    int kw = *(static_cast<const int *>(fields[2].data));
    int sh = *(static_cast<const int *>(fields[3].data));
    int sw = *(static_cast<const int *>(fields[4].data));
    int ph = *(static_cast<const int *>(fields[5].data));
    int pw = *(static_cast<const int *>(fields[6].data));
    int deformableGroup = *(static_cast<const int *>(fields[7].data));
    int i_n = *(static_cast<const int *>(fields[8].data));
    int i_c = *(static_cast<const int *>(fields[9].data));
    int i_h = *(static_cast<const int *>(fields[10].data));
    int i_w = *(static_cast<const int *>(fields[11].data));
    int o_n = *(static_cast<const int *>(fields[12].data));
    int o_c = *(static_cast<const int *>(fields[13].data));
    int o_h = *(static_cast<const int *>(fields[14].data));
    int o_w = *(static_cast<const int *>(fields[15].data));
    std::vector<dnnType> mask_v(static_cast<const dnnType*>(fields[16].data),static_cast<const dnnType*>(fields[16].data)+fields[16].length);
    std::vector<dnnType> offset_v(static_cast<const dnnType*>(fields[17].data),static_cast<const dnnType*>(fields[17].data)+fields[17].length);
    std::vector<dnnType> ones_d2_v(static_cast<const dnnType*>(fields[18].data),static_cast<const dnnType*>(fields[18].data)+fields[18].length);
    std::vector<dnnType> ones_d1_v(static_cast<const dnnType*>(fields[19].data),static_cast<const dnnType*>(fields[19].data)+fields[19].length);
    std::vector<dnnType> data_d_v(static_cast<const dnnType*>(fields[20].data),static_cast<const dnnType*>(fields[20].data)+fields[20].length);
    std::vector<dnnType> bias2_d_v(static_cast<const dnnType*>(fields[21].data),static_cast<const dnnType*>(fields[21].data)+fields[21].length);
    int height_ones = *(static_cast<const int *>(fields[22].data));
    int width_ones = *(static_cast<const int *>(fields[23].data));
    int dim_ones = *(static_cast<const int *>(fields[24].data));
    auto *pluginObj = new DeformableConvRT(chunk_dim,kh,kw,sh,sw,ph,pw,deformableGroup,i_n,i_c,i_h,i_w,o_n,o_c,o_h,o_w,data_d_v,bias2_d_v,ones_d1_v,ones_d2_v,offset_v,mask_v,height_ones,width_ones,dim_ones);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *DeformableConvRTPluginCreator::getPluginName() const NOEXCEPT {
    return DEFORMABLECONVRT_PLUGIN_NAME;
}

const char *DeformableConvRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return DEFORMABLECONVRT_PLUGIN_VERSION;
}

const PluginFieldCollection *DeformableConvRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}





