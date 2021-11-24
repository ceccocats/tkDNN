#include "NvInfer.h"
#include "../kernels.h"
#include <cassert>
#include <vector>

namespace nvinfer1 {
    class ActivationLeakyRT : public IPluginV2 {

    public:
        explicit ActivationLeakyRT(float s);

        ActivationLeakyRT(const void *data, size_t length);

        ~ActivationLeakyRT();

        int getNbOutputs() const NOEXCEPT override;

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) NOEXCEPT override;

        void
        configureWithFormat(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, DataType type,
                            PluginFormat format, int maxBatchSize) NOEXCEPT override;

        int initialize() NOEXCEPT override;

        void terminate() NOEXCEPT override {}

        size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override;

#if NV_TENSORRT_MAJOR > 7
        int enqueue(int batchSize, void const *const *inputs, void *const *outputs, void *workspace,
                    cudaStream_t stream) NOEXCEPT override;
#elif NV_TENSORRT_MAJOR == 7
        int32_t enqueue (int32_t batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;
#endif

        size_t getSerializationSize() const NOEXCEPT override;

        void serialize(void *buffer) const NOEXCEPT override;

        bool supportsFormat(DataType type, PluginFormat format) const NOEXCEPT override;

        const char *getPluginType() const NOEXCEPT override;

    REGISTER_TENSORRT_PLUGIN(ActivationLeakyRTPluginCreator);
};