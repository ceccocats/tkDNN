#include <tkDNN/pluginsRT/YoloRT.h>

#include <utility>
using namespace nvinfer1;

std::vector<PluginField> YoloRTPluginCreator::mPluginAttributes;
PluginFieldCollection YoloRTPluginCreator::mFC{};

static const char* YOLORT_PLUGIN_VERSION{"1"};
static const char* YOLORT_PLUGIN_NAME{"YoloRT_tkDNN"};

YoloRT::YoloRT(int classes, int num, int c,int h,int w,std::vector<std::string> classNames,
               std::vector<float> masks_v,std::vector<float> bias_v,int n_masks, float scale_xy,
               float nms_thresh, int nms_kind,
               int new_coords) {
    this->c = c;
    this->h = h;
    this->w = w;
    this->classes = classes;
    this->num = num;
    this->n_masks = n_masks;
    this->scaleXY = scale_xy;
    this->nms_thresh = nms_thresh;
    this->nms_kind = nms_kind;
    this->new_coords = new_coords;
    this->classesNames = std::move(classNames);
    this->mask = std::move(masks_v);
    this->bias = std::move(bias_v);

}

YoloRT::YoloRT(const void *data, size_t length) {
    std::vector<float> maskTemp,biasTemp;
    const char* buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
    classes = readBUF<int>(buf);
    num = readBUF<int>(buf);
    n_masks = readBUF<int>(buf);
    scaleXY = readBUF<float>(buf);
    nms_thresh = readBUF<float>(buf);
    nms_kind = readBUF<int>(buf);
    new_coords = readBUF<int>(buf);
    c = readBUF<int>(buf);
    h = readBUF<int>(buf);
    w = readBUF<int>(buf);
    mask.resize(n_masks);
    for(int i=0;i<n_masks;i++){
        mask[i] = readBUF<dnnType>(buf);
    }
    bias.resize(n_masks*2*num);
    for(int i=0;i<n_masks*2*num;i++){
        bias[i] = readBUF<dnnType>(buf);
    }
    classesNames.resize(classes);
    for(int i=0;i<classes;i++){
        char tmp[YOLORT_CLASSNAME_W];
        for(int j=0;j<YOLORT_CLASSNAME_W;j++)
            tmp[j] = readBUF<char>(buf);
        classesNames[1] = std::string(tmp);
    }
    assert(buf == bufCheck + length);
}

YoloRT::~YoloRT() {}

int YoloRT::getNbOutputs() const NOEXCEPT {
    return 1;
}

Dims YoloRT::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT {
    return inputs[0];
}



int YoloRT::initialize() NOEXCEPT {
    return 0;
}

void YoloRT::terminate() NOEXCEPT {}

size_t YoloRT::getWorkspaceSize(int maxBatchSize) const NOEXCEPT {
    return 0;
}

#if NV_TENSORRT_MAJOR > 7
int YoloRT::enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) NOEXCEPT {
    dnnType *srcData = (dnnType *) reinterpret_cast<const dnnType *>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType *>(outputs[0]);

    checkCuda(cudaMemcpyAsync(dstData, srcData, batchSize * c * h * w * sizeof(dnnType), cudaMemcpyDeviceToDevice,
                              stream));


    for (int b = 0; b < batchSize; ++b) {
        for (int n = 0; n < n_masks; ++n) {
            int index = entry_index(b, n * w * h, 0);
            if (new_coords == 1) {
                if (this->scaleXY != 1)
                    scalAdd(dstData + index, 2 * w * h, this->scaleXY, -0.5 * (this->scaleXY - 1), 1);
            } else {
                activationLOGISTICForward(srcData + index, dstData + index, 2 * w * h, stream); //x,y

                if (this->scaleXY != 1)
                    scalAdd(dstData + index, 2 * w * h, this->scaleXY, -0.5 * (this->scaleXY - 1), 1);

                index = entry_index(b, n * w * h, 4);
                activationLOGISTICForward(srcData + index, dstData + index, (1 + classes) * w * h, stream);
            }
        }
    }

    //std::cout<<"YOLO END\n";
    return 0;
}
#elif NV_TENSORRT_MAJOR == 7
int32_t YoloRT::enqueue(int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    dnnType *srcData = (dnnType *) reinterpret_cast<const dnnType *>(inputs[0]);
    dnnType *dstData = reinterpret_cast<dnnType *>(outputs[0]);

    checkCuda(cudaMemcpyAsync(dstData, srcData, batchSize * c * h * w * sizeof(dnnType), cudaMemcpyDeviceToDevice,
                              stream));


    for (int b = 0; b < batchSize; ++b) {
        for (int n = 0; n < n_masks; ++n) {
            int index = entry_index(b, n * w * h, 0);
            if (new_coords == 1) {
                if (this->scaleXY != 1)
                    scalAdd(dstData + index, 2 * w * h, this->scaleXY, -0.5 * (this->scaleXY - 1), 1);
            } else {
                activationLOGISTICForward(srcData + index, dstData + index, 2 * w * h, stream); //x,y

                if (this->scaleXY != 1)
                    scalAdd(dstData + index, 2 * w * h, this->scaleXY, -0.5 * (this->scaleXY - 1), 1);

                index = entry_index(b, n * w * h, 4);
                activationLOGISTICForward(srcData + index, dstData + index, (1 + classes) * w * h, stream);
            }
        }
    }

    //std::cout<<"YOLO END\n";
    return 0;
}
#endif


size_t YoloRT::getSerializationSize() const NOEXCEPT {
    return 8 * sizeof(int) + 2 * sizeof(float) + n_masks * sizeof(dnnType) + num * n_masks * 2 * sizeof(dnnType) +
           YOLORT_CLASSNAME_W * classes * sizeof(char);
}

bool YoloRT::supportsFormat(DataType type, PluginFormat format) const NOEXCEPT {
    return true;
}

void YoloRT::serialize(void *buffer) const NOEXCEPT {
    char *buf = reinterpret_cast<char *>(buffer), *a = buf;
    writeBUF(buf, classes);    //std::cout << "Classes :" << classes << std::endl;
    writeBUF(buf, num);        //std::cout << "Num : " << num << std::endl;
    writeBUF(buf, n_masks);    //std::cout << "N_Masks" << n_masks << std::endl;
    writeBUF(buf, scaleXY);    //std::cout << "ScaleXY :" << scaleXY << std::endl;
    writeBUF(buf, nms_thresh); //std::cout << "nms_thresh :" << nms_thresh << std::endl;
    writeBUF(buf, nms_kind);    //std::cout << "nms_kind : " << nms_kind << std::endl;
    writeBUF(buf, new_coords); //std::cout << "new_coords : " << new_coords << std::endl;
    writeBUF(buf, c);            //std::cout << "C : " << c << std::endl;
    writeBUF(buf, h);            //std::cout << "H : " << h << std::endl;
    writeBUF(buf, w);            //std::cout << "C : " << c << std::endl;
    for (int i = 0; i < n_masks; i++) {
        writeBUF(buf, mask[i]); //std::cout << "mask[i] : " << mask[i] << std::endl;
    }
    for (int i = 0; i < n_masks * 2 * num; i++) {
        writeBUF(buf, bias[i]); //std::cout << "bias[i] : " << bias[i] << std::endl;
    }

    // save classes names
    for (int i = 0; i < classes; i++) {
        char tmp[YOLORT_CLASSNAME_W];
        strcpy(tmp, classesNames[i].c_str());
        for (int j = 0; j < YOLORT_CLASSNAME_W; j++) {
            writeBUF(buf, tmp[j]);
        }
    }
    assert(buf == a + getSerializationSize());
}

const char *YoloRT::getPluginType() const NOEXCEPT {
    return YOLORT_PLUGIN_NAME;
}

const char *YoloRT::getPluginVersion() const NOEXCEPT {
    return YOLORT_PLUGIN_VERSION;
}

void YoloRT::destroy() NOEXCEPT {
    delete this;
}

const char *YoloRT::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

void YoloRT::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

IPluginV2Ext *YoloRT::clone() const NOEXCEPT {
    auto *p = new YoloRT(classes, num,c,h,w,classesNames,mask,bias, n_masks, scaleXY, nms_thresh, nms_kind, new_coords);
    p->setPluginNamespace(mPluginNamespace.c_str());
    return p;
}

DataType YoloRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const NOEXCEPT {
    return DataType::kFLOAT;
}

void YoloRT::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                             IGpuAllocator *gpuAllocator) NOEXCEPT {

}

void YoloRT::configurePlugin(const Dims *inputDims, int32_t nbInputs, const Dims *outputDims, int32_t nbOutputs,
                             const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                             const bool *outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT {

}

bool YoloRT::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const NOEXCEPT {
    return false;
}

bool YoloRT::canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT {
    return false;
}

void YoloRT::detachFromContext() NOEXCEPT {

}

YoloRTPluginCreator::YoloRTPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void YoloRTPluginCreator::setPluginNamespace(const char *pluginNamespace) NOEXCEPT {
    mPluginNamespace = pluginNamespace;
}

const char *YoloRTPluginCreator::getPluginNamespace() const NOEXCEPT {
    return mPluginNamespace.c_str();
}

IPluginV2Ext *YoloRTPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT {
    auto *pluginObj = new YoloRT(serialData,serialLength);
    pluginObj->setPluginNamespace(mPluginNamespace.c_str());
    return pluginObj;
}

IPluginV2Ext *YoloRTPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT {
    const PluginField *fields = fc->fields;
    int classes = *(static_cast<const int *>(fields[0].data));
    int num = *(static_cast<const int *>(fields[1].data));
    int c = *(static_cast<const int *>(fields[2].data));
    int h = *(static_cast<const int *>(fields[3].data));
    int w = *(static_cast<const int *>(fields[4].data));
    std::vector<std::string> classNames(static_cast<const std::string *>(fields[5].data),static_cast<const std::string *>(fields[5].data) + fields[5].length);
    std::vector<dnnType> mask_v(static_cast<const dnnType*>(fields[6].data),static_cast<const dnnType*>(fields[6].data) + fields[6].length);
    std::vector<dnnType> bias_v(static_cast<const dnnType*>(fields[7].data),static_cast<const dnnType*>(fields[7].data) + fields[7].length);
    int n_masks = *(static_cast<const int *>(fields[8].data));
    dnnType scaleXY = *(static_cast<const float*>(fields[9].data));
    dnnType nmsThresh = *(static_cast<const float*>(fields[10].data));
    int nms_kind = *(static_cast<const int*>(fields[11].data));
    int new_coords = *(static_cast<const int*>(fields[12].data));
    auto *pluginObj = new YoloRT(classes,num,c,h,w,classNames,mask_v,bias_v,n_masks,scaleXY,nmsThresh,nms_kind,new_coords);
    return pluginObj;
}

const char *YoloRTPluginCreator::getPluginName() const NOEXCEPT {
    return YOLORT_PLUGIN_NAME;
}

const char *YoloRTPluginCreator::getPluginVersion() const NOEXCEPT {
    return YOLORT_PLUGIN_VERSION;
}

const PluginFieldCollection *YoloRTPluginCreator::getFieldNames() NOEXCEPT {
    return &mFC;
}














