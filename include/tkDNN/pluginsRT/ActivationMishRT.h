#include<cassert>
#include "../kernels.h"

class ActivationMishRT : public IPluginV2 {

public:
    ActivationMishRT() {}

    ~ActivationMishRT() {}

    ActivationMishRT(const void *data, size_t length) {
        std::cout<<"DESERIALIZE MISH"<<std::endl;
        const char *buf = reinterpret_cast<const char *>(data), *bufCheck = buf;
        size = readBUF<int>(buf);
        assert(buf == bufCheck + length);
    }


    int getNbOutputs() const NOEXCEPT override { return 1; }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override { return inputs[0]; }

    void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                             PluginFormat format, int maxBatchSize) NOEXCEPT override {
        assert(format == PluginFormat::kLINEAR);
        size = 1;
        for (int i = 0; i < outputDims[0].nbDims; i++)
            size *= outputDims[0].d[i];
    }

    int initialize() NOEXCEPT override { return 0; }

    virtual void terminate() NOEXCEPT override {}

    virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override { return 0; }

    virtual int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) NOEXCEPT override {
        activationMishForward((dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
                              reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, stream);
        return 0;
    }


    virtual size_t getSerializationSize() const NOEXCEPT override {
        return 1 * sizeof(int);
    }

    virtual void serialize(void *buffer) const NOEXCEPT override {
        char *buf = reinterpret_cast<char *>(buffer), *a = buf;
        tk::dnn::writeBUF(buf, size);
        assert(buf == a + getSerializationSize());
    }

    const char *getPluginType() const NOEXCEPT override {
        return "ActivationMishRT_tkDNN";
    }

    const char *getPluginVersion() const NOEXCEPT override {
        return "1";
    }

    void destroy() NOEXCEPT override { delete this; }

    bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
        return true;
    }

    const char *getPluginNamespace() const NOEXCEPT override {
        return mPluginNamespace.c_str();
    }

    void setPluginNamespace(const char *plguinNamespace) NOEXCEPT override {
        mPluginNamespace = plguinNamespace;
    }

    IPluginV2 *clone() const NOEXCEPT override {
        ActivationMishRT *p = new ActivationMishRT();
        p->setPluginNamespace(mPluginNamespace.c_str());
        return p;
    }

    int size;
private:
    std::string mPluginNamespace;
};

class ActivationMishRTPluginCreator : public IPluginCreator {
public:
    ActivationMishRTPluginCreator() {
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override {
        mPluginNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }

    IPluginV2 *deserializePlugin(const char* name,const void* serialData,size_t serialLength) NOEXCEPT override{
        ActivationMishRT *pluginObj = new ActivationMishRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char* name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        ActivationMishRT *pluginObj = new ActivationMishRT();
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "ActivationMishRT_tkDNN";
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

REGISTER_TENSORRT_PLUGIN(ActivationMishRTPluginCreator);