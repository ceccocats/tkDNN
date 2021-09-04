#ifndef _UPSAMPLERT_PLUGIN_H
#define _UPSAMPLERT_PLUGIN_H

#include<cassert>
#include "../kernels.h"
#include <NvInfer.h>
#include <vector>

namespace nvinfer1 {

    class UpsampleRT : public IPluginV2 {

    public:
        explicit UpsampleRT(int stride);

        UpsampleRT(const void *data, size_t length);

        ~UpsampleRT();

        int getNbOutputs() const NOEXCEPT override;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override;

        void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                            PluginFormat format, int maxBatchSize) NOEXCEPT override;

        int initialize() NOEXCEPT override;

        void terminate() NOEXCEPT override;

        size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override;

        int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                            cudaStream_t stream) NOEXCEPT override;


        size_t getSerializationSize() const NOEXCEPT override;

        void serialize(void *buffer) const NOEXCEPT override;

        bool supportsFormat(DataType type, PluginFormat format) const NOEXCEPT override;

        const char *getPluginType() const NOEXCEPT override;

        const char *getPluginVersion() const NOEXCEPT override;

        void destroy() NOEXCEPT override;

        const char *getPluginNamespace() const NOEXCEPT override;

        void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override;

        IPluginV2 *clone() const NOEXCEPT override ;

            int c, h, w, stride;
            private:
            std::string mPluginNamespace;
        };

        class UpsampleRTPluginCreator : public IPluginCreator {
        public:
            UpsampleRTPluginCreator();

            void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override;

            const char *getPluginNamespace() const NOEXCEPT override;

            IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT override;

            IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT override;

            const char *getPluginName() const NOEXCEPT override;

            const char *getPluginVersion() const NOEXCEPT override;

            const PluginFieldCollection *getFieldNames() NOEXCEPT override;

        private:
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
            std::string mPluginNamespace;
        };

        REGISTER_TENSORRT_PLUGIN(UpsampleRTPluginCreator);
    };

#endif