#ifndef RESIZE_LAYER_RT_H
#define RESIZE_LAYER_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"

#define PLUGIN_NAME "Resize"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class ResizeLayerRT final : public nvinfer1::IPluginV2 {

public:
	ResizeLayerRT(int c, int h, int w) {
		o_c = c;
		o_h = h;
		o_w = w;
	}

	~ResizeLayerRT() = default;

	int getNbOutputs() const noexcept override {
		return 1;
	}

	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override {
		return nvinfer1::Dims3{o_c, o_h, o_w};
	}

	void configureWithFormat(nvinfer1::Dims const * inputDims,
					int32_t nbInputs,
					nvinfer1::Dims const * outputDims,
					int32_t nbOutputs,
					nvinfer1::DataType type,
					nvinfer1::PluginFormat format,
					int32_t maxBatchSize) noexcept override {
		i_c = inputDims[0].d[0];
		i_h = inputDims[0].d[1];
		i_w = inputDims[0].d[2];
	}

	int initialize() noexcept override {
		return 0;
	}

	void terminate() noexcept override {
	}

	size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
		return 0;
	}

	int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
    	// printf("%d %d %d %d %d %d\n", i_c, i_w, i_h, o_c, o_w, o_h);
        resizeForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
					  reinterpret_cast<dnnType*>(outputs[0]),
					  batchSize, i_c, i_h, i_w, o_c, o_h, o_w, stream);
		return 0;
	}

	size_t getSerializationSize() const noexcept override {
		return 6*sizeof(int);
	}

	void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;

		writeBUF(buf, o_c);
		writeBUF(buf, o_h);
		writeBUF(buf, o_w);

		writeBUF(buf, i_c);
		writeBUF(buf, i_h);
		writeBUF(buf, i_w);
		assert(buf == a + getSerializationSize());
	}

	// Extra IPluginV2 overrides
	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
			return true;
	}

	nvinfer1::IPluginV2 * clone() const noexcept override {
			auto a = new ResizeLayerRT(*this);
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
	int i_c, i_h, i_w, o_c, o_h, o_w;
};

class ResizeLayerRTCreator final : public nvinfer1::IPluginCreator {
public:
    ResizeLayerRTCreator() = default;

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
			int o_cTemp = readBUF<int>(buf);
			int o_hTemp = readBUF<int>(buf);
			int o_wTemp = readBUF<int>(buf);
			ResizeLayerRT* r = new ResizeLayerRT(o_cTemp, o_hTemp, o_wTemp);

			r->i_c = readBUF<int>(buf);
			r->i_h = readBUF<int>(buf);
			r->i_w = readBUF<int>(buf);
			assert(buf == bufCheck + serialLength);
			return r;
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

#endif // RESIZE_LAYER_RT_H