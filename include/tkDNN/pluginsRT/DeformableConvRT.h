#ifndef _DEFORMABLECONVRT_PLUGIN_H
#define _DEFORMABLECONVRT_PLUGIN_H

#include <NvInfer.h>
#include <vector>
#include<cassert>
#include "../kernels.h"
#include <tkdnn.h>

namespace nvinfer1 {
    class DeformableConvRT : public IPluginV2Ext {


    public:
        DeformableConvRT(int chunk_dim, int kh, int kw, int sh, int sw, int ph, int pw,
                         int deformableGroup, int i_n, int i_c, int i_h, int i_w,
                         int o_n, int o_c, int o_h, int o_w,std::vector<dnnType> data_H,std::vector<dnnType> bias2_H,
                         std::vector<dnnType> ones_d1_h,std::vector<dnnType> ones_d2_h,std::vector<dnnType> offsetH,std::vector<dnnType> maskH,int height_ones,
                         int width_ones,int dim_ones);

        ~DeformableConvRT();

        DeformableConvRT(const void *data, size_t length) ;

        int getNbOutputs() const NOEXCEPT override ;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override ;

        int initialize() NOEXCEPT override ;

        void terminate() NOEXCEPT override ;

        size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override ;
#if NV_TENSORRT_MAJOR > 7
        int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                            cudaStream_t stream) NOEXCEPT override;
#elif NV_TENSORRT_MAJOR == 7
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


        cublasStatus_t stat;
        cublasHandle_t handle{nullptr};
        int i_n, i_c, i_h, i_w;
        int o_n, o_c, o_h, o_w;
        int size;
        int chunk_dim;
        int kh, kw;
        int sh, sw;
        int ph, pw;
        int deformableGroup;
        int height_ones;
        int width_ones;
        int dim_ones;

        std::vector<dnnType> data_d_v;
        std::vector<dnnType> bias2_d_v;
        std::vector<dnnType> ones_d1_v;
        std::vector<dnnType> offset_v;
        std::vector<dnnType> mask_v;
        std::vector<dnnType> ones_d2_v;
        dnnType* data_d;
        dnnType* bias2_d;
        dnnType* ones_d1;
        dnnType* offset;
        dnnType* mask;
        dnnType* ones_d2;
        // dnnType *input_n;
        // dnnType *offset_n;
        // dnnType *mask_n;
        // dnnType *output_n;


        tk::dnn::DeformConv2d *defRT;

    private:
        std::string mPluginNamespace;
    };

    class DeformableConvRTPluginCreator : public IPluginCreator {
    public:
        TKDNN_LIB_EXPORT_API DeformableConvRTPluginCreator();

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
    REGISTER_TENSORRT_PLUGIN(DeformableConvRTPluginCreator);
};
#endif
