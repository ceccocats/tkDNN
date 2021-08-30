#include<cassert>
#include <vector>
#include "../kernels.h"
#define YOLORT_CLASSNAME_W 256


class YoloRT : public IPluginV2 {

public:
    YoloRT(int classes, int num, tk::dnn::Yolo *Yolo = nullptr, int n_masks = 3, float scale_xy = 1,
           float nms_thresh = 0.45, int nms_kind = 0, int new_coords = 0) {
        this->yolo = Yolo;
        this->classes = classes;
        this->num = num;
        this->n_masks = n_masks;
        this->scaleXY = scale_xy;
        this->nms_thresh = nms_thresh;
        this->nms_kind = nms_kind;
        this->new_coords = new_coords;

        mask = new dnnType[n_masks];
        bias = new dnnType[num * n_masks * 2];
        if (yolo != nullptr) {
            memcpy(mask, yolo->mask_h, sizeof(dnnType) * n_masks);
            memcpy(bias, yolo->bias_h, sizeof(dnnType) * num * n_masks * 2);
            classesNames = yolo->classesNames;
        }
    }

    YoloRT(const void *data,size_t length){
        std::vector<float> maskTemp,biasTemp;
        std::cout<<"LENGTH : "<<length<<std::endl;
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
        for(int i=0;i<n_masks;i++){
            maskTemp.push_back(readBUF<dnnType>(buf));
            std::cout<<maskTemp[i]<<std::endl;
        }
        for(int i=0;i<n_masks*2*num;i++){
            biasTemp.push_back(readBUF<dnnType>(buf));
            std::cout<<biasTemp[i]<<std::endl;
        }
        mask = maskTemp.data();
        bias = biasTemp.data();
        classesNames.resize(classes);
        for(int i=0;i<classes;i++){
            char tmp[YOLORT_CLASSNAME_W];
            for(int j=0;j<YOLORT_CLASSNAME_W;j++)
                tmp[j] = readBUF<char>(buf);
            classesNames[1] = std::string(tmp);
        }
        assert(buf == bufCheck + length);

    }

    ~YoloRT() {

    }



    int getNbOutputs() const NOEXCEPT override {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override {
        return inputs[0];
    }

    void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                             PluginFormat format, int maxBatchSize) NOEXCEPT override {
        c = inputDims[0].d[0];
        h = inputDims[0].d[1];
        w = inputDims[0].d[2];
    }

    int initialize() NOEXCEPT override {

        return 0;
    }

    virtual void terminate() NOEXCEPT override {
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {
        return 0;
    }

    virtual int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) NOEXCEPT override {

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


    virtual size_t getSerializationSize() const NOEXCEPT override {
        return 8 * sizeof(int) + 2 * sizeof(float) + n_masks * sizeof(dnnType) + num * n_masks * 2 * sizeof(dnnType) +
               YOLORT_CLASSNAME_W * classes * sizeof(char);
    }

    bool supportsFormat(DataType type, PluginFormat format) const NOEXCEPT override {
        return true; //todo implement proper supportsFormat
    }

    virtual void serialize(void *buffer) const NOEXCEPT override {
        char *buf = reinterpret_cast<char *>(buffer), *a = buf;
        tk::dnn::writeBUF(buf, classes);    //std::cout << "Classes :" << classes << std::endl;
        tk::dnn::writeBUF(buf, num);        //std::cout << "Num : " << num << std::endl;
        std::cout<<num<<std::endl;
        tk::dnn::writeBUF(buf, n_masks);    //std::cout << "N_Masks" << n_masks << std::endl;
        tk::dnn::writeBUF(buf, scaleXY);    //std::cout << "ScaleXY :" << scaleXY << std::endl;
        tk::dnn::writeBUF(buf, nms_thresh); //std::cout << "nms_thresh :" << nms_thresh << std::endl;
        tk::dnn::writeBUF(buf, nms_kind);    //std::cout << "nms_kind : " << nms_kind << std::endl;
        tk::dnn::writeBUF(buf, new_coords); //std::cout << "new_coords : " << new_coords << std::endl;
        tk::dnn::writeBUF(buf, c);            //std::cout << "C : " << c << std::endl;
        tk::dnn::writeBUF(buf, h);            //std::cout << "H : " << h << std::endl;
        tk::dnn::writeBUF(buf, w);            //std::cout << "C : " << c << std::endl;
        for (int i = 0; i < n_masks; i++) {
            tk::dnn::writeBUF(buf, mask[i]); //std::cout << "mask[i] : " << mask[i] << std::endl;
        }
        for (int i = 0; i < n_masks * 2 * num; i++) {
            tk::dnn::writeBUF(buf, bias[i]); //std::cout << "bias[i] : " << bias[i] << std::endl;
        }

        // save classes names
        for (int i = 0; i < classes; i++) {
            char tmp[YOLORT_CLASSNAME_W];
            strcpy(tmp, classesNames[i].c_str());
            for (int j = 0; j < YOLORT_CLASSNAME_W; j++) {
                tk::dnn::writeBUF(buf, tmp[j]);
            }
        }
        assert(buf == a + getSerializationSize());
    }

    const char *getPluginType() const NOEXCEPT override {
        return "YoloRT_tkDNN";
    }

    const char *getPluginVersion() const NOEXCEPT override {
        return "1";
    }

    void destroy() NOEXCEPT override { delete this; }

    const char *getPluginNamespace() const NOEXCEPT override {
        return mPluginNamespace.c_str();
    }

    void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override {
        mPluginNamespace = pluginNamespace;
    }

    IPluginV2 *clone() const NOEXCEPT override {
        YoloRT *p = new YoloRT(classes, num,yolo, n_masks, scaleXY, nms_thresh, nms_kind, new_coords);
        p->setPluginNamespace(mPluginNamespace.c_str());
        return p;
    }

    Yolo *yolo;
    int c, h, w;
    int classes, num, n_masks;
    float scaleXY;
    float nms_thresh;
    int nms_kind;
    int new_coords;
    int NUM=0;
    std::vector<std::string> classesNames;

    dnnType *mask;
    dnnType *bias;

    int entry_index(int batch, int location, int entry) {
        int n = location / (w * h);
        int loc = location % (w * h);
        return batch * c * h * w + n * w * h * (4 + classes + 1) + entry * w * h + loc;
    }

private:
    std::string mPluginNamespace;

};

class YoloRTPluginCreator : public IPluginCreator{
public:
    YoloRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("classes",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("num",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("yolo",nullptr,PluginFieldType::kUNKNOWN,1));
        mPluginAttributes.emplace_back(PluginField("numMasks",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("scaleXY",nullptr,PluginFieldType::kFLOAT32,1));
        mPluginAttributes.emplace_back(PluginField("nmsThresh",nullptr,PluginFieldType::kFLOAT32,1));
        mPluginAttributes.emplace_back(PluginField("nmsKind",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("newCoords",nullptr,PluginFieldType::kINT32,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }

    IPluginV2 *deserializePlugin(const char *name,const void *serialData,size_t serialLength) NOEXCEPT override{
        YoloRT *pluginObj = new YoloRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char* name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        //todo assert
        int classes = *(static_cast<const int *>(fields[0].data));
        int num = *(static_cast<const int *>(fields[1].data));
        Yolo *yoloTemp = const_cast<Yolo *>(static_cast<const Yolo *>(fields[2].data));
        int numMasks = *(static_cast<const int*>(fields[3].data));
        float scaleXY = *(static_cast<const float *>(fields[4].data));
        float nmsThresh = *(static_cast<const float *>(fields[5].data));
        int nmsKind = *(static_cast<const int *>(fields[6].data));
        int newCoords = *(static_cast<const int *>(fields[7].data));
        YoloRT *pluginObj = new YoloRT(classes,num,yoloTemp,numMasks,scaleXY,nmsThresh,nmsKind,newCoords);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "YoloRT_tkDNN";
    }

    const char *getPluginVersion() const NOEXCEPT override{
        return "1";
    }

    const PluginFieldCollection *getFieldNames() NOEXCEPT override{
        return &mFC;
    }

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};

REGISTER_TENSORRT_PLUGIN(YoloRTPluginCreator);
