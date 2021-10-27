#ifndef ROUTE_RT_H
#define ROUTE_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"

#define PLUGIN_NAME "Route"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class RouteRT final : public nvinfer1::IPluginV2 {

	/**
		THIS IS NOT USED ANYMORE
	*/

public:
	RouteRT(int groups, int group_id) {
		this->groups = groups;
		this->group_id = group_id;
	}

	~RouteRT() = default;

	int getNbOutputs() const noexcept override {
		return 1;
	}

	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override {
		int out_c = 0;
		for(int i=0; i<nbInputDims; i++) out_c += inputs[i].d[0];
		return nvinfer1::Dims3{out_c/groups, inputs[0].d[1], inputs[0].d[2]};
	}

	void configureWithFormat(nvinfer1::Dims const * inputDims,
					int32_t nbInputs,
					nvinfer1::Dims const * outputDims,
					int32_t nbOutputs,
					nvinfer1::DataType type,
					nvinfer1::PluginFormat format,
					int32_t maxBatchSize) noexcept override {
		in = nbInputs;
		c = 0;
		for(int i=0; i<nbInputs; i++) {
			c_in[i] = inputDims[i].d[0];
			c += inputDims[i].d[0];
		}
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
		c /= groups;
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
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		for(int b=0; b<batchSize; b++) {
			int offset = 0;
			for(int i=0; i<in; i++) {
				dnnType *input = (dnnType*)reinterpret_cast<const dnnType*>(inputs[i]);
				int in_dim = c_in[i]*h*w;
				int part_in_dim = in_dim / this->groups;
				checkCuda( cudaMemcpyAsync(dstData + b*c*w*h + offset, input + b*c*w*h*groups + this->group_id*part_in_dim, part_in_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream) );
				offset += part_in_dim;
			}
		}

		return 0;
	}


	size_t getSerializationSize() const noexcept override {
		return (6+MAX_INPUTS)*sizeof(int);
	}

	void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		writeBUF(buf, groups);
		writeBUF(buf, group_id);
		writeBUF(buf, in);
		for(int i=0; i<MAX_INPUTS; i++)
			writeBUF(buf, c_in[i]);

		writeBUF(buf, c);
		writeBUF(buf, h);
		writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

	// Extra IPluginV2 overrides
	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
			return true;
	}

	nvinfer1::IPluginV2 * clone() const noexcept override {
			auto a = new RouteRT(*this);
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

	static const int MAX_INPUTS = 4;
	int in;
	int c_in[MAX_INPUTS];
	int c, h, w;
	int groups, group_id;
};

class RouteRTCreator final : public nvinfer1::IPluginCreator {
public:
    RouteRTCreator() = default;

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
			int groupsTemp = readBUF<int>(buf);
			int group_idTemp = readBUF<int>(buf);
			RouteRT* r = new RouteRT(groupsTemp, group_idTemp);
			r->in = readBUF<int>(buf);
			for(int i=0; i<RouteRT::MAX_INPUTS; i++)
					r->c_in[i] = readBUF<int>(buf);
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

#endif // ROUTE_RT_H