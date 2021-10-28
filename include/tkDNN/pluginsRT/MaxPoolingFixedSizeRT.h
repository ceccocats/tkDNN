#include<cassert>
#include "../kernels.h"
#include <NvInfer.h>
#include <vector>
#include <utils.h>


namespace nvinfer1 {
    class MaxPoolFixedSizeRT : public IPluginV2Ext {

    public:
        MaxPoolFixedSizeRT(int c, int h, int w, int n, int strideH, int strideW, int winSize, int padding) ;

        MaxPoolFixedSizeRT(const void *data, size_t length) ;

        ~MaxPoolFixedSizeRT() ;

        int getNbOutputs() const NOEXCEPT override ;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override ;

        int initialize() NOEXCEPT override ;

        void terminate() NOEXCEPT override ;

        size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override ;

#if NV_TENSORRT_MAJOR > 7
        int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                            cudaStream_t stream) NOEXCEPT override ;
#elif NV_TENSORRT_MAJOR <= 7
        int32_t enqueue (int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;
#endif


        size_t getSerializationSize() const NOEXCEPT override ;

        void serialize(void *buffer) const NOEXCEPT override ;

        void destroy() NOEXCEPT override ;

        bool supportsFormat(DataType type, PluginFormat format) const NOEXCEPT override ;

        const char *getPluginNamespace() const NOEXCEPT override ;

        void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override ;

        const char *getPluginType() const NOEXCEPT override ;

        const char *getPluginVersion() const NOEXCEPT override ;

        IPluginV2Ext *clone() const NOEXCEPT override ;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const NOEXCEPT override;

        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) NOEXCEPT override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const NOEXCEPT override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT override;

        void configurePlugin (Dims const *inputDims, int32_t nbInputs, Dims const *outputDims,
                              int32_t nbOutputs, DataType const *inputTypes, DataType const *outputTypes,
                              bool const *inputIsBroadcast, bool const *outputIsBroadcast, PluginFormat floatFormat,
                              int32_t maxBatchSize) NOEXCEPT override;

        void detachFromContext() NOEXCEPT override;


        int n, c, h, w;
        int stride_H, stride_W;
        int winSize;
        int padding;

    private:
        std::string mPluginNamespace;
    };

    class MaxPoolFixedSizeRTPluginCreator : public IPluginCreator {
    public:
        MaxPoolFixedSizeRTPluginCreator() ;

        void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override ;

        const char *getPluginNamespace() const NOEXCEPT override ;

        IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT override ;

        IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT override ;

        const char *getPluginName() const NOEXCEPT override ;

        const char *getPluginVersion() const NOEXCEPT override ;

        const PluginFieldCollection *getFieldNames() NOEXCEPT override ;

    private:
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
        std::string mPluginNamespace;

    };

    REGISTER_TENSORRT_PLUGIN(MaxPoolFixedSizeRTPluginCreator);
};
