#ifndef FLATTEN_CONCAT_RT_H
#define FLATTEN_CONCAT_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"

#define PLUGIN_NAME "Flatten"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class FlattenConcatRT final : public nvinfer1::IPluginV2 {

public:
	FlattenConcatRT() {
		stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf ("CUBLAS initialization failed\n");
			return;
  		}
	}

	~FlattenConcatRT() = default;

	int getNbOutputs() const noexcept override {
		return 1;
	}

	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override {
		return nvinfer1::Dims3{ inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2], 1, 1};
	}

	void configureWithFormat(nvinfer1::Dims const * inputDims,
					int32_t nbInputs,
					nvinfer1::Dims const * outputDims,
					int32_t nbOutputs,
					nvinfer1::DataType type,
					nvinfer1::PluginFormat format,
					int32_t maxBatchSize) noexcept override {
		assert(nbOutputs == 1 && nbInputs ==1);
		rows = inputDims[0].d[0];
		cols = inputDims[0].d[1] * inputDims[0].d[2];
		c = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
		h = 1;
		w = 1;
	}

	int initialize() noexcept override {
		return 0;
	}

	void terminate() noexcept override {
		checkERROR(cublasDestroy(handle));
	}

	size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
		return 0;
	}

	int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*rows*cols*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

		checkERROR( cublasSetStream(handle, stream) );
		for(int i=0; i<batchSize; i++) {
			float const alpha(1.0);
			float const beta(0.0);
			int offset = i*rows*cols;
			checkERROR( cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, srcData + offset, cols, &beta, srcData + offset, rows, dstData + offset, rows ));
		}
		return 0;
	}

	size_t getSerializationSize() const noexcept override {
		return 5*sizeof(int);
	}

	void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a = buf;
		writeBUF(buf, c);
		writeBUF(buf, h);
		writeBUF(buf, w);
		writeBUF(buf, rows);
		writeBUF(buf, cols);
		assert(buf == a + getSerializationSize());
	}

	// Extra IPluginV2 overrides
	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
			return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR);
	}

	nvinfer1::IPluginV2 * clone() const noexcept override {
			auto a = new FlattenConcatRT(*this);
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

	int c, h, w;
	int rows, cols;
	cublasStatus_t stat;
	cublasHandle_t handle;
};

class FlattenConcatRTCreator final : public nvinfer1::IPluginCreator {
public:
    FlattenConcatRTCreator() = default;

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
			FlattenConcatRT *r = new FlattenConcatRT();
			r->c = readBUF<int>(buf);
			r->h = readBUF<int>(buf);
			r->w = readBUF<int>(buf);
			r->rows = readBUF<int>(buf);
			r->cols = readBUF<int>(buf);
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

#endif // FLATTEN_CONCAT_RT_H