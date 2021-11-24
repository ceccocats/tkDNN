#ifndef _SHORTCUTRT_PLUGIN_H
#define _SHORTCUTRT_PLUGIN_H

#include<cassert>
#include "../kernels.h"
#include <NvInfer.h>
#include <vector>
#include <tkdnn.h>


namespace nvinfer1 {

    class ShortcutRT : public IPluginV2Ext {

    public:
        ShortcutRT(int bc,int bh,int bw,int c,int h,int w ,bool mul);

        ~ShortcutRT();

        ShortcutRT(const void *data, size_t length);

        int getNbOutputs() const NOEXCEPT override;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override;

        void configurePlugin (Dims const *inputDims, int32_t nbInputs, Dims const *outputDims, int32_t nbOutputs,
                              DataType const *inputTypes, DataType const *outputTypes, bool const *inputIsBroadcast,
                              bool const *outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) NOEXCEPT override;

        bool isOutputBroadcastAcrossBatch (int32_t outputIndex, bool const *inputIsBroadcasted, int32_t nbInputs) const NOEXCEPT override;

        bool canBroadcastInputAcrossBatch (int32_t inputIndex) const NOEXCEPT override;

        void attachToContext (cudnnContext *, cublasContext *, IGpuAllocator *) NOEXCEPT override;

        void detachFromContext () NOEXCEPT override;

        DataType getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const NOEXCEPT override;

        int initialize() NOEXCEPT override;

        void terminate() NOEXCEPT override;

        size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override;

#if NV_TENSORRT_MAJOR > 7
        int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) NOEXCEPT override;
#elif NV_TENSORRT_MAJOR == 7
        int32_t enqueue (int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;
#endif


        size_t getSerializationSize() const NOEXCEPT override;

        void serialize(void *buffer) const NOEXCEPT override;

        bool supportsFormat(DataType type, PluginFormat format) const NOEXCEPT override;

        const char *getPluginType() const NOEXCEPT override;

        const char *getPluginVersion() const NOEXCEPT override;

        void destroy() NOEXCEPT override;

        const char *getPluginNamespace() const NOEXCEPT override;

        void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override;

        IPluginV2Ext *clone() const NOEXCEPT override;

        int c, h, w;
        int bc, bh, bw,bl;
        bool mul;
        tk::dnn::dataDim_t bDim;
    private:
        std::string mPluginNamespace;
    };


    class ShortcutRTPluginCreator : public IPluginCreator {
    public:
        ShortcutRTPluginCreator();

        void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override;

        const char *getPluginNamespace() const NOEXCEPT override;

        IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT override;

        IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT override;

        const char *getPluginName() const NOEXCEPT override;

        const char *getPluginVersion() const NOEXCEPT override;

        const PluginFieldCollection *getFieldNames() NOEXCEPT override;

    public:
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
        std::string mPluginNamespace;
    };

    REGISTER_TENSORRT_PLUGIN(ShortcutRTPluginCreator);

};

#endif