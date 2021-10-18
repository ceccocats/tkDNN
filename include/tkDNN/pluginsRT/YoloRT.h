#ifndef _YOLORT_PLUGIN_H
#define _YOLORT_PLUGIN_H

#include<cassert>
#include <vector>
#include "../kernels.h"
#include <NvInfer.h>
#include <tkdnn.h>

#define YOLORT_CLASSNAME_W 256

namespace nvinfer1 {
    class YoloRT : public IPluginV2 {

    public:
        YoloRT(int classes, int num, tk::dnn::Yolo *Yolo = nullptr, int n_masks = 3, float scale_xy = 1,
               float nms_thresh = 0.45, int nms_kind = 0, int new_coords = 0);

        YoloRT(const void *data, size_t length);

        ~YoloRT();


        int getNbOutputs() const NOEXCEPT override;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override;

        void configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                            PluginFormat format, int maxBatchSize) NOEXCEPT override;

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

        bool supportsFormat(DataType type, PluginFormat format) const NOEXCEPT override;

        void serialize(void *buffer) const NOEXCEPT override;

        const char *getPluginType() const NOEXCEPT override;

        const char *getPluginVersion() const NOEXCEPT override;

        void destroy() NOEXCEPT override;

        const char *getPluginNamespace() const NOEXCEPT override;

        void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override;

        IPluginV2 *clone() const NOEXCEPT override;

        tk::dnn::Yolo *yolo;
        int c, h, w;
        int classes, num, n_masks;
        float scaleXY;
        float nms_thresh;
        int nms_kind;
        int new_coords;
        int NUM = 0;
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

    class YoloRTPluginCreator : public IPluginCreator {
    public:
        YoloRTPluginCreator();

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

    REGISTER_TENSORRT_PLUGIN(YoloRTPluginCreator);
};
#endif