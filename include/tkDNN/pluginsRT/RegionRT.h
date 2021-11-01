#ifndef REGION_RT_H
#define REGION_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"

#define PLUGIN_NAME "Region"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class RegionRT final : public nvinfer1::IPluginV2 {

public:
	RegionRT(int classes, int coords, int num) {
		this->classes = classes;
		this->coords = coords;
		this->num = num;
	}

	~RegionRT() = default;

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
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
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
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

		for (int b = 0; b < batchSize; ++b){
			for(int n = 0; n < num; ++n){
				int index = entry_index(b, n*w*h, 0);
				activationLOGISTICForward(srcData + index, dstData + index, 2*w*h, stream);

				index = entry_index(b, n*w*h, coords);
				activationLOGISTICForward(srcData + index, dstData + index, w*h, stream);
			}
		}

		//softmax start
		int index = entry_index(0, 0, coords + 1);
		softmaxForward(	srcData + index, classes, batchSize*num,
						(c*h*w)/num,
						w*h, 1, w*h, 1, dstData + index, stream);

		return 0;
	}


	size_t getSerializationSize() const noexcept override {
		return 6*sizeof(int);
	}

	void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		writeBUF(buf, classes);
		writeBUF(buf, coords);
		writeBUF(buf, num);
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
			auto a = new RegionRT(*this);
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
  int classes, coords, num;

	int entry_index(int batch, int location, int entry) {
		int n =   location / (w*h);
		int loc = location % (w*h);
		return batch*c*h*w + n*w*h*(coords+classes+1) + entry*w*h + loc;
	}

};

class RegionRTCreator final : public nvinfer1::IPluginCreator {
public:
    RegionRTCreator() = default;

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
			int classesTemp = readBUF<int>(buf);
			int coordsTemp = readBUF<int>(buf);
			int numTemp = readBUF<int>(buf);
			RegionRT* r = new RegionRT(classesTemp, coordsTemp, numTemp);

			r->c = readBUF<int>(buf);
			r->h = readBUF<int>(buf);
			r->w = readBUF<int>(buf);
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

#endif // REGION_RT_H