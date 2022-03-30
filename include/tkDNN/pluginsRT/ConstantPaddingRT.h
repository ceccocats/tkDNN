//
// Created by perseusdg on 1/7/22.
//

#ifndef _CONSTANTPADDINGRT_PLUGIN_H
#define _CONSTANTPADDINGRT_PLUGIN_H

#include<cassert>
#include <NvInfer.h>
#include <vector>
#include <utils.h>
#include <kernels.h>

namespace nvinfer1{
    class ConstantPaddingRT : public IPluginV2Ext {
    public:
        ConstantPaddingRT(int32_t padH,int32_t padW,int32_t n,int32_t c,int32_t i_h,int32_t i_w,int32_t o_h,int32_t o_w,float constant);

        ConstantPaddingRT(const void *data,size_t length);

        ~ConstantPaddingRT();

        int getNbOutputs() const NOEXCEPT override;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override ;

        int initialize() NOEXCEPT override ;

        void terminate() NOEXCEPT override ;

        size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override ;


#if NV_TENSORRT_MAJOR > 7
        int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) NOEXCEPT override ;
#elif NV_TENSORRT_MAJOR <= 7
        int32_t enqueue (int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;
#endif

        size_t getSerializationSize() const NOEXCEPT override ;

        void serialize(void *buffer) const NOEXCEPT override ;

        void destroy() NOEXCEPT override ;

        const char *getPluginType() const NOEXCEPT override ;

        const char *getPluginVersion() const NOEXCEPT override;

        const char *getPluginNamespace() const NOEXCEPT override ;

        void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override ;

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

        bool supportsFormat (DataType type, PluginFormat format) const NOEXCEPT override;

        int32_t i_h,i_w,o_h,o_w,n,c,padH,padW;
        float constant;
    private:
        std::string mPluginNamespace;

    };

    class ConstantPaddingRTPluginCreator : public IPluginCreator {
    public:
        ConstantPaddingRTPluginCreator();

        void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override;

        const char *getPluginNamespace() const NOEXCEPT override;

        IPluginV2Ext *deserializePlugin(const char *name, const void *serialData, size_t serialLength) NOEXCEPT override ;

        IPluginV2Ext *createPlugin(const char *name, const PluginFieldCollection *fc) NOEXCEPT override ;

        const char *getPluginName() const NOEXCEPT override ;

        const char *getPluginVersion() const NOEXCEPT override;

        const PluginFieldCollection *getFieldNames() NOEXCEPT override ;

    private:
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
        std::string mPluginNamespace;

    };

    REGISTER_TENSORRT_PLUGIN(ConstantPaddingRTPluginCreator);
};


#endif //TKDNN_CONSTANTPADDINGRT_H
