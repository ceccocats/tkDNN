#include<cassert>
#include "../kernels.h"

class ReorgRT : public IPluginV2 {

public:
	ReorgRT(int stride) {
		this->stride = stride;
	}

	~ReorgRT(){

	}

	ReorgRT(const void* data,size_t length){
	    const char* buf  = reinterpret_cast<const char*>(data),*bufCheck = buf;
	    stride = readBUF<int>(buf);
	    c = readBUF<int>(buf);
	    h = readBUF<int>(buf);
	    w = readBUF<int>(buf);
	    assert(buf == bufCheck + length);
	}

	int getNbOutputs() const NOEXCEPT override {return 1;}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{inputs[0].d[0]*stride*stride, inputs[0].d[1]/stride, inputs[0].d[2]/stride};
	}

	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format, int maxBatchSize) NOEXCEPT override {
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
	}

	int initialize() NOEXCEPT override { return 0;}

	virtual void terminate() NOEXCEPT override {}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override { return 0;}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {

		reorgForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
					  reinterpret_cast<dnnType*>(outputs[0]), 
					  batchSize, c, h, w, stride, stream);
		return 0;
	}


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 4*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, stride);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}
	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{return true;}

	const char *getPluginType() const NOEXCEPT override{
	    return "ReorgRT_tkDNN";
	}

	const char* getPluginVersion() const NOEXCEPT override{
	    return "1";
	}
	void destroy() NOEXCEPT override{ delete this;}

	const char* getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}

	IPluginV2* clone() const NOEXCEPT override{
	    ReorgRT *p = new ReorgRT(stride);
	    p->setPluginNamespace(mPluginNamespace.c_str());
        return p;
	}

	int c, h, w, stride;
private:
    std::string mPluginNamespace;
};

class ReorgRTPluginCreator : public IPluginCreator{
public:
    ReorgRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("stride",nullptr,PluginFieldType::kINT32,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }

    IPluginV2* deserializePlugin(const char* name,const void* serialData,size_t serialLength) NOEXCEPT override{
        ReorgRT *pluginObj = new ReorgRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char* name,const PluginFieldCollection* fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        assert(fc->nbFields == 1);
        assert(fields[0].type == PluginFieldType::kINT32);
        int stride = *(static_cast<const int *>(fields[0].data));
        ReorgRT *pluginObj = new ReorgRT(stride);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "ReorgRT_tkDNN";
    }

    const char *getPluginVersion() const NOEXCEPT override{
        return "1";
    }

    const PluginFieldCollection *getFieldNames() NOEXCEPT override{
        return &mFC;
    }
private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};

REGISTER_TENSORRT_PLUGIN(ReorgRTPluginCreator);

