#include <tkDNN/pluginsRT/DeformableConvRT.h>
using namespace nvinfer1;
using namespace tk::dnn;

std::vector<PluginField> DeformableConvRTPluginCreator::mPluginAttributes;
PluginFieldCollection DeformableConvRTPluginCreator::mFC{};


DeformableConvRT::DeformableConvRT(int chunk_dim, int kh, int kw, int sh, int sw, int ph, int pw, int deformableGroup,
                                   int i_n, int i_c, int i_h, int i_w, int o_n, int o_c, int o_h, int o_w,
                                   tk::dnn::DeformConv2d *deformable) {
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
    this->defRT = deformable;

    height_ones = (i_h + 2 * ph - (1 * (kh - 1) + 1)) / sh + 1;
    width_ones = (i_w + 2 * pw - (1 * (kw - 1) + 1)) / sw + 1;
    dim_ones = i_c * kh * kw * 1 * height_ones * width_ones;

    checkCuda( cudaMalloc(&data_d, i_c * o_c * kh * kw * 1 * sizeof(dnnType)));
    checkCuda( cudaMalloc(&bias2_d, o_c*sizeof(dnnType)));
    checkCuda( cudaMalloc(&ones_d1, height_ones * width_ones * sizeof(dnnType)));
    checkCuda( cudaMalloc(&offset, 2*chunk_dim*sizeof(dnnType)));
    checkCuda( cudaMalloc(&mask, chunk_dim*sizeof(dnnType)));
    checkCuda( cudaMalloc(&ones_d2, dim_ones*sizeof(dnnType)));
    if(deformable != nullptr) {
        checkCuda( cudaMemcpy(data_d, deformable->data_d, sizeof(dnnType)*i_c * o_c * kh * kw * 1, cudaMemcpyDeviceToDevice) );
        checkCuda( cudaMemcpy(bias2_d, deformable->bias2_d, sizeof(dnnType)*o_c, cudaMemcpyDeviceToDevice) );
        checkCuda( cudaMemcpy(ones_d1, deformable->ones_d1, sizeof(dnnType)*height_ones*width_ones, cudaMemcpyDeviceToDevice) );
        checkCuda( cudaMemcpy(offset, deformable->offset, sizeof(dnnType)*2*chunk_dim, cudaMemcpyDeviceToDevice) );
        checkCuda( cudaMemcpy(mask, deformable->mask, sizeof(dnnType)*chunk_dim, cudaMemcpyDeviceToDevice) );
        checkCuda( cudaMemcpy(ones_d2, deformable->ones_d2, sizeof(dnnType)*dim_ones, cudaMemcpyDeviceToDevice) );
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
        FatalError("CUBLAS initialization failed\n");

}

DeformableConvRT::~DeformableConvRT() {
    checkCuda( cudaFree(data_d) );
    checkCuda( cudaFree(bias2_d) );
    checkCuda( cudaFree(ones_d1) );
    checkCuda( cudaFree(offset) );
    checkCuda( cudaFree(mask) );
    checkCuda( cudaFree(ones_d2) );
    cublasDestroy(handle);
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
    dnnType *aus = new dnnType[chunk_dim*2];
    for(int i=0;i<chunk_dim*2;i++)
        aus[i] = readBUF<dnnType>(buf);
    checkCuda(cudaMemcpy(offset,aus,sizeof(dnnType)*2*chunk_dim,cudaMemcpyHostToDevice));
    free(aus);

    aus = new dnnType[chunk_dim];
    for(int i=0;i<chunk_dim;i++)
        aus[i] = readBUF<dnnType>(buf);
    checkCuda(cudaMemcpy(mask,aus,sizeof(dnnType)*chunk_dim,cudaMemcpyHostToDevice));
    free(aus);

    aus = new dnnType[i_c*o_c*kh*kw*1];
    for(int i=0;i<(i_c*o_c*kh*kw*1);i++)
        aus[i] = readBUF<dnnType>(buf);
    checkCuda(cudaMemcpy(data_d,aus,sizeof(dnnType)*(i_c*o_c*kh*kw*1),cudaMemcpyHostToDevice));
    free(aus);

    aus = new dnnType[o_c];
    for(int i=0; i < o_c; i++)
        aus[i] = readBUF<dnnType>(buf);
    checkCuda( cudaMemcpy(bias2_d, aus, sizeof(dnnType)*o_c, cudaMemcpyHostToDevice) );
    free(aus);

    aus = new dnnType[height_ones * width_ones];
    for(int i=0; i<height_ones * width_ones; i++)
        aus[i] = readBUF<dnnType>(buf);
    checkCuda( cudaMemcpy(ones_d1, aus, sizeof(dnnType)*height_ones * width_ones, cudaMemcpyHostToDevice) );
    free(aus);

    aus = new dnnType[dim_ones];
    for(int i=0; i<dim_ones; i++)
        aus[i] = readBUF<dnnType>(buf);
    checkCuda( cudaMemcpy(ones_d2, aus, sizeof(dnnType)*dim_ones, cudaMemcpyHostToDevice) );
    free(aus);

    assert(buf == bufCheck + length);

}

int DeformableConvRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims DeformableConvRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return Dims3{defRT->output_dim.c, defRT->output_dim.h, defRT->output_dim.w};
}

void DeformableConvRT::configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,DataType type, PluginFormat format, int maxBatchSize) NOEXCEPT {}

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
#elif NV_TENSORRT_MAJOR == 7
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
    return 16 * sizeof(int) + chunk_dim * 3 * sizeof(dnnType) + (i_c * o_c * kh * kw * 1 ) * sizeof(dnnType) +
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
    dnnType *aus = new dnnType[chunk_dim*2];
    checkCuda( cudaMemcpy(aus, offset, sizeof(dnnType)*2*chunk_dim, cudaMemcpyDeviceToHost) );
    for(int i=0; i<chunk_dim*2; i++)
        writeBUF(buf, aus[i]);
    free(aus);
    aus = new dnnType[chunk_dim];
    checkCuda( cudaMemcpy(aus, mask, sizeof(dnnType)*chunk_dim, cudaMemcpyDeviceToHost) );
    for(int i=0; i<chunk_dim; i++)
        writeBUF(buf, aus[i]);
    free(aus);
    aus = new dnnType[(i_c * o_c * kh * kw * 1 )];
    checkCuda( cudaMemcpy(aus, data_d, sizeof(dnnType)*(i_c * o_c * kh * kw * 1 ), cudaMemcpyDeviceToHost) );
    for(int i=0; i<(i_c * o_c * kh * kw * 1 ); i++)
        writeBUF(buf, aus[i]);
    free(aus);
    aus = new dnnType[o_c];
    checkCuda( cudaMemcpy(aus, bias2_d, sizeof(dnnType)*o_c, cudaMemcpyDeviceToHost) );
    for(int i=0; i < o_c; i++)
        writeBUF(buf, aus[i]);
    free(aus);
    aus = new dnnType[height_ones * width_ones];
    checkCuda( cudaMemcpy(aus, ones_d1, sizeof(dnnType)*height_ones * width_ones, cudaMemcpyDeviceToHost) );
    for(int i=0; i<height_ones * width_ones; i++)
        writeBUF(buf, aus[i]);
    free(aus);
    aus = new dnnType[dim_ones];
    checkCuda( cudaMemcpy(aus, ones_d2, sizeof(dnnType)*dim_ones, cudaMemcpyDeviceToHost) );
    for(int i=0; i<dim_ones; i++)
        writeBUF(buf, aus[i]);
    free(aus);
    assert(buf == a + getSerializationSize());
}

void DeformableConvRT::destroy() NOEXCEPT { delete this; }

bool DeformableConvRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
    //todo assert
}

const char *DeformableConvRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void DeformableConvRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *DeformableConvRT::getPluginType() const NOEXCEPT {
    return "DeformableConvRT_tkDNN";
}

const char *DeformableConvRT::getPluginVersion() const NOEXCEPT {
    return "1";
}

IPluginV2 *DeformableConvRT::clone() const NOEXCEPT {
    auto *p = new DeformableConvRT(chunk_dim,kh,kw,sh,sw,ph,pw,deformableGroup,i_n,i_c,i_h,i_w,o_n,o_c,o_h,o_w,defRT);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
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

IPluginV2 *DeformableConvRTPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                            size_t serialLength) NOEXCEPT {
    auto *pluginObj = new DeformableConvRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2 *DeformableConvRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
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
    int o_w = *(static_cast<const int *>(fields[14].data));
    auto *defRT = const_cast<DeformConv2d *>(static_cast<const DeformConv2d *>(fields[15].data));
    auto *pluginObj = new DeformableConvRT(chunk_dim,kh,kw,sh,sw,ph,pw,deformableGroup,i_n,i_c,i_h,i_w,o_n,o_c,o_h,o_w,defRT);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

const char *DeformableConvRTPluginCreator::getPluginName() const NOEXCEPT {
    return "DeformableConvRT_tkDNN";
}

const char *DeformableConvRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return "1";
}

const PluginFieldCollection *DeformableConvRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}





