#ifndef ACTIVATION_LOGISTIC_RT_H
#define ACTIVATION_LOGISTIC_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"

#define PLUGIN_NAME "ActivationLogistic"
#define PLUGIN_VERSION "1"

namespace tk { namespace dnn {

class ActivationLogisticRT final : public nvinfer1::IPluginV2 {

public:
    ActivationLogisticRT() = default;

    ~ActivationLogisticRT() = default;

    int getNbOutputs() const noexcept override {
        return 1;
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override {
        return inputs[0];
    }

    void configureWithFormat(nvinfer1::Dims const * inputDims,
            int32_t nbInputs,
            nvinfer1::Dims const * outputDims,
            int32_t nbOutputs,
            nvinfer1::DataType type,
            nvinfer1::PluginFormat format,
            int32_t maxBatchSize) noexcept override {
        size = 1;
        for(int i=0; i<outputDims[0].nbDims; i++)
            size *= outputDims[0].d[i];
    }

    int initialize() noexcept override {
        return 0;
    }

    void terminate() noexcept override {
    }

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int32_t enqueue(int32_t batchSize, const void* const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        activationLOGISTICForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
                                  reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, stream);
        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return 1*sizeof(int);
    }

    void serialize(void* buffer) const noexcept override {
        char *buf = reinterpret_cast<char*>(buffer);
        writeBUF(buf, size);
    }

    // Extra IPluginV2 overrides
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
        return true;
    }

    nvinfer1::IPluginV2 * clone() const noexcept override {
        auto a = new ActivationLogisticRT(*this);
        return a;
    }

    const char* getPluginType() const noexcept override {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override {
        return PLUGIN_VERSION;
    }

    void destroy() noexcept override {}

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

    std::string mNamespace;
    int size;
};

class ActivationLogisticRTCreator final : public nvinfer1::IPluginCreator {
public:
    ActivationLogisticRTCreator() = default;

    const char* getPluginName() const noexcept override {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override {
        return PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &mFC;
    }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        std::cout << "Create plugin" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        const char * buf = reinterpret_cast<const char*>(serialData),*bufCheck = buf;
        ActivationLogisticRT *a = new ActivationLogisticRT();
        a->size = readBUF<int>(buf);
        assert(buf == bufCheck + serialLength);
        return a;
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
}}
#undef PLUGIN_NAME
#undef PLUGIN_VERSION

#endif // ACTIVATION_LOGISTIC_RT_H