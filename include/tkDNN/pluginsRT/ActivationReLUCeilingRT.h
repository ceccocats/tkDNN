#ifndef ACTIVATION_RELU_CEILING_RT_H
#define ACTIVATION_RELU_CEILING_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"

#define PLUGIN_NAME "ActivationCReLU"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class ActivationReLUCeiling final : public nvinfer1::IPluginV2 {

public:
	ActivationReLUCeiling(const float ceiling) {
		this->ceiling = ceiling;
	}

	~ActivationReLUCeiling() = default;

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
		activationReLUCeilingForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
											reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, ceiling, stream);
		return 0;
	}

	size_t getSerializationSize() const noexcept override {
		return 1*sizeof(int) +  1*sizeof(float);
	}

	void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		writeBUF(buf, ceiling);
		writeBUF(buf, size);
		assert(buf = a + getSerializationSize());

	}

	// Extra IPluginV2 overrides
	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
			return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR);
	}

	nvinfer1::IPluginV2 * clone() const noexcept override {
			auto a = new ActivationReLUCeiling(*this);
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
	float ceiling;
};

class ActivationReLUCeilingCreator final : public nvinfer1::IPluginCreator {
public:
    ActivationReLUCeilingCreator() = default;

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
			float activationReluTemp = readBUF<float>(buf);
			ActivationReLUCeiling* a = new ActivationReLUCeiling(activationReluTemp);
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

#endif // ACTIVATION_RELU_CEILING_RT_H