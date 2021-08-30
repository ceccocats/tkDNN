#include<cassert>
#include "../kernels.h"

class ActivationLogisticRT : public IPluginV2 {

public:
    ActivationLogisticRT() {

    }

    ActivationLogisticRT(const void *data, size_t length)
    {
        const char* buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
        size = readBUF<int>(buf);
        assert(buf == bufCheck + length);

    }

    ~ActivationLogisticRT(){

    }

    int getNbOutputs() const NOEXCEPT override {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT  override {
        return inputs[0];
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format, int maxBatchSize) NOEXCEPT override {
        size = 1;
        for(int i=0; i<outputDims[0].nbDims; i++)
            size *= outputDims[0].d[i];
    }

    int initialize() NOEXCEPT override {

        return 0;
    }

    virtual void terminate() NOEXCEPT override {
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {

        activationLOGISTICForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
                                  reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, stream);
        return 0;
    }


    virtual size_t getSerializationSize() const NOEXCEPT override {
        return 1*sizeof(int);
    }

    virtual void serialize(void* buffer) const NOEXCEPT override {
        char *buf = reinterpret_cast<char*>(buffer);
        tk::dnn::writeBUF(buf, size);
    }

    const char *getPluginType() const NOEXCEPT override {
        return "ActivationLogisticRT_tkDNN";
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

    bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
        return true;
        //todo assert;
    }

    IPluginV2 *clone() const NOEXCEPT override{
        ActivationLogisticRT *p = new ActivationLogisticRT();
        p->setPluginNamespace(mPluginNamespace.c_str());
        return p;
    }

    int size;

private:
    std::string mPluginNamespace;
};

class ActivationLogisticRTPluginCreator : public IPluginCreator{
public:
    ActivationLogisticRTPluginCreator(){
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT override {
        ActivationLogisticRT *pluginObj = new ActivationLogisticRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginNamespace() const NOEXCEPT override {
        return mPluginNamespace.c_str();
    }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT override {
        ActivationLogisticRT *pluginObj = new ActivationLogisticRT();
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return  pluginObj;
    }

    const char *getPluginVersion() const NOEXCEPT override{
        return "1";
    }

    const PluginFieldCollection *getFieldNames() NOEXCEPT override{
        return &mFC;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "ActivationLogisticRT_tkDNN";
    }

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};

REGISTER_TENSORRT_PLUGIN(ActivationLogisticRTPluginCreator);