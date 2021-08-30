#include<cassert>
#include "../kernels.h"


class UpsampleRT : public IPluginV2 {

public:
	UpsampleRT(int stride) {
		this->stride = stride;
	}

	UpsampleRT(const void *data,size_t length){
	    const char* buf = reinterpret_cast<const char*>(data),*bufCheck=buf;
	    stride = readBUF<int>(buf);
	    c = readBUF<int>(buf);
	    h = readBUF<int>(buf);
	    w = readBUF<int>(buf);
	    assert(buf == bufCheck + length);
	}


	~UpsampleRT(){}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3(inputs[0].d[0], inputs[0].d[1]*stride, inputs[0].d[2]*stride);
	}

	void configureWithFormat (const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format,int maxBatchSize) NOEXCEPT override {
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
	}

	int initialize() NOEXCEPT override {return 0;}

	virtual void terminate() NOEXCEPT override {}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {	return 0;}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {

		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
	    
		fill(dstData, batchSize*c*h*w*stride*stride, 0.0, stream);
    	upsampleForward(srcData, dstData, batchSize, c, h, w, stride, 1, 1, stream);
		return 0;
	}


	virtual size_t getSerializationSize() const NOEXCEPT override {	return 4*sizeof(int);}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, stride);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
	    //todo assert
	    return true;
	}

	const char *getPluginType() const NOEXCEPT override{
	    return "1";
	}

	const char *getPluginVersion() const NOEXCEPT override{
	    static const char* UPSAMPLE_RT_PLUGIN = "UpsampleRT_TRT";
	    return UPSAMPLE_RT_PLUGIN;
	}

	void destroy() NOEXCEPT override{delete this;}

	const char *getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}

	IPluginV2* clone() const NOEXCEPT override{
	    UpsampleRT *p = new UpsampleRT(stride);
	    p->setPluginNamespace(mPluginNamespace.c_str());
	    return p;
	}

	int c, h, w, stride;
private:
    std::string mPluginNamespace;
};

class UpsampleRTPluginCreator : public IPluginCreator{
public:
    UpsampleRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("stride",nullptr,PluginFieldType::kINT32,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }

    IPluginV2 *deserializePlugin(const char* name,const void* serialData,size_t serialLength) NOEXCEPT override{
        UpsampleRT *pluginObj = new UpsampleRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char* name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        int stride = *(static_cast<const int *>(fields[0].data));
        UpsampleRT *pluginObj = new UpsampleRT(stride);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        static const char* UPSAMPLE_RT_PLUGIN = "UpsampleRT_TRT";
        return UPSAMPLE_RT_PLUGIN;
    }

    const char *getPluginVersion() const NOEXCEPT override{
        return "1";
    }

    const PluginFieldCollection *getFieldNames() NOEXCEPT override{
        return &mFC;
    }
private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};

REGISTER_TENSORRT_PLUGIN(UpsampleRTPluginCreator);

