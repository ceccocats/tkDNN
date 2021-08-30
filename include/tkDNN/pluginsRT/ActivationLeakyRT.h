#include "../kernels.h"
#include <cassert>

class ActivationLeakyRT : public IPluginV2 {

public:
    ActivationLeakyRT(float s) { slope = s; }

    ActivationLeakyRT(const void *data, size_t length)
    {
        std::cout<<"DESERIALIZE LEAKYRT"<<std::endl;
        const char* buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
        slope = readBUF<float>(buf);
        size = readBUF<int>(buf);
        assert(buf == bufCheck + length);

    }

    ~ActivationLeakyRT() {}

    int getNbOutputs() const NOEXCEPT override { return 1; }

    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override {
        return inputs[0];
    }

    void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,DataType type,PluginFormat format, int maxBatchSize) NOEXCEPT override
    {
        assert(type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
        size = 1;
        for (int i = 0; i < outputDims[0].nbDims; i++)
            size *= outputDims[0].d[i];
    }

    int initialize() NOEXCEPT override { return 0; }

    virtual void terminate() NOEXCEPT override {}

    virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {
        return 0;
    }

    virtual int enqueue(int batchSize, void const *const *inputs, void *const *outputs, void *workspace,
                        cudaStream_t stream) NOEXCEPT override {
        activationLEAKYForward(
                (dnnType *) reinterpret_cast<const dnnType *>(inputs[0]),
                reinterpret_cast<dnnType *>(outputs[0]), batchSize * size, slope,
                stream);
        return 0;
    }

    virtual size_t getSerializationSize() const NOEXCEPT override {
        return 1 * sizeof(int) + 1 * sizeof(float);
    }

    virtual void serialize(void *buffer) const NOEXCEPT override {
        char *buf = reinterpret_cast<char *>(buffer), *a = buf;
        tk::dnn::writeBUF(buf, size);
        assert(buf == a + getSerializationSize());
    }

    bool supportsFormat(DataType type, PluginFormat format) const NOEXCEPT override {
        return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
    }

    const char *getPluginType() const NOEXCEPT override {
        return "ActivationLeakyRT_tkDNN";
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

    IPluginV2* clone() const NOEXCEPT override {
        ActivationLeakyRT *p = new ActivationLeakyRT(slope);
        p->setPluginNamespace(mPluginNamespace.c_str());
        return p;
    }

    int size;
    float slope;

private:
    std::string mPluginNamespace;
};

class ActivationLeakyRTPluginCreator : public IPluginCreator {
public:
    ActivationLeakyRTPluginCreator() {
        mPluginAttributes.emplace_back(
                PluginField("slope", nullptr, PluginFieldType::kFLOAT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT override {
        ActivationLeakyRT *pluginObj = new ActivationLeakyRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginNamespace() const NOEXCEPT override {
        return mPluginNamespace.c_str();
    }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT override {
        const PluginField *fields = fc->fields;
        assert(fc->nbFields == 1);
        assert(fields[0].type == PluginFieldType::kFLOAT32);
        float slope = *(static_cast<const float *>(fields[0].data));
        ActivationLeakyRT *pluginObj = new ActivationLeakyRT(slope);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return  pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "ActivationLeakyRT_tkDNN";
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

REGISTER_TENSORRT_PLUGIN(ActivationLeakyRTPluginCreator);