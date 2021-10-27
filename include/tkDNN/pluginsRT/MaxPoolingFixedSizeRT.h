#ifndef MAX_POOLING_FIXED_SIZE_RT_H
#define MAX_POOLING_FIXED_SIZE_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"

#define PLUGIN_NAME "Pooling"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class MaxPoolFixedSizeRT final : public nvinfer1::IPluginV2 {

public:
	MaxPoolFixedSizeRT(int c, int h, int w, int n, int strideH, int strideW, int winSize, int padding) {
		this->c = c;
		this->h = h;
		this->w = w;
		this->n = n;
		this->stride_H = strideH;
		this->stride_W = strideW;
		this->winSize = winSize;
		this->padding = padding;
	}

	~MaxPoolFixedSizeRT() = default;

	int getNbOutputs() const noexcept override {
		return 1;
	}

	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override {
		return nvinfer1::Dims3{this->c, this->h, this->w};
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

	void terminate() noexcept override {
	}

	size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
		return 0;
	}

	int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
		//std::cout<<this->n<<"  "<<this->c<<"  "<<this->h<<"  "<<this->w<<"  "<<this->stride_H<<"  "<<this->stride_W<<"  "<<this->winSize<<"  "<<this->padding<<std::endl;
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		MaxPoolingForward(srcData, dstData, batchSize, this->c, this->h, this->w, this->stride_H, this->stride_W, this->winSize, this->padding, stream);
		return 0;
	}

	size_t getSerializationSize() const noexcept override {
		return 8*sizeof(int);
	}

	void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;

		writeBUF(buf, this->c);
		writeBUF(buf, this->h);
		writeBUF(buf, this->w);
		writeBUF(buf, this->n);
		writeBUF(buf, this->stride_H);
		writeBUF(buf, this->stride_W);
		writeBUF(buf, this->winSize);
		writeBUF(buf, this->padding);
		assert(buf == a + getSerializationSize());
	}

	// Extra IPluginV2 overrides
	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
			return true;
	}

	nvinfer1::IPluginV2 * clone() const noexcept override {
			auto a = new MaxPoolFixedSizeRT(*this);
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
	int stride_H, stride_W;
	int winSize;
	int padding;
};

class MaxPoolFixedSizeRTCreator final : public nvinfer1::IPluginCreator {
public:
    MaxPoolFixedSizeRTCreator() = default;

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
			int cTemp = readBUF<int>(buf);
			int hTemp = readBUF<int>(buf);
			int wTemp = readBUF<int>(buf);
			int nTemp = readBUF<int>(buf);
			int strideHTemp = readBUF<int>(buf);
			int strideWTemp = readBUF<int>(buf);
			int winSizeTemp = readBUF<int>(buf);
			int paddingTemp = readBUF<int>(buf);

			MaxPoolFixedSizeRT* r = new MaxPoolFixedSizeRT(cTemp, hTemp, wTemp, nTemp, strideHTemp, strideWTemp, winSizeTemp, paddingTemp);
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

#endif // MAX_POOLING_FIXED_SIZE_RT_H