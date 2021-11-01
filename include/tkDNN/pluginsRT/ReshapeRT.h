#ifndef RESHAPE_RT_H
#define RESHAPE_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"
#include "../Network.h"

#define PLUGIN_NAME "Reshape"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class ReshapeRT final : public nvinfer1::IPluginV2 {

public:
	ReshapeRT(dataDim_t new_dim) {
		n = new_dim.n;
		c = new_dim.c;
		h = new_dim.h;
		w = new_dim.w;
	}

	~ReshapeRT() = default;

	int getNbOutputs() const noexcept override {
		return 1;
	}

	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override {
		return nvinfer1::Dims3{ c,h,w};
	}

	void configureWithFormat(nvinfer1::Dims const * inputDims,
					int32_t nbInputs,
					nvinfer1::Dims const * outputDims,
					int32_t nbOutputs,
					nvinfer1::DataType type,
					nvinfer1::PluginFormat format,
					int32_t maxBatchSize) noexcept override {
	}

	int initialize() noexcept override {
		return 0;
	}

	virtual void terminate() noexcept override {
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
		return 0;
	}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
		return 0;
	}

	virtual size_t getSerializationSize() const noexcept override {
		return 4*sizeof(int);
	}

	virtual void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a = buf;
		writeBUF(buf, n);
		writeBUF(buf, c);
		writeBUF(buf, h);
		writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

	// Extra IPluginV2 overrides
	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
			return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR);
	}

	nvinfer1::IPluginV2 * clone() const noexcept override {
			auto a = new ReshapeRT(*this);
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
	int n, c, h, w;
};

class ReshapeRTCreator final : public nvinfer1::IPluginCreator {
public:
    ReshapeRTCreator() = default;

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
			dataDim_t new_dim;
			new_dim.n = readBUF<int>(buf);
			new_dim.c = readBUF<int>(buf);
			new_dim.h = readBUF<int>(buf);
			new_dim.w = readBUF<int>(buf);
			ReshapeRT *r = new ReshapeRT(new_dim);
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

#endif // RESHAPE_RT_H