#include<cassert>

class ReshapeRT : public IPluginV2 {

public:
	ReshapeRT(dataDim_t newDim) {
	    new_dim = newDim;
		n = new_dim.n;
		c = new_dim.c;
		h = new_dim.h;
		w = new_dim.w;
	}

	ReshapeRT(const void *data,size_t length){
	    const char *buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
	    new_dim.n = readBUF<int>(buf);
	    new_dim.c = readBUF<int>(buf);
	    new_dim.h = readBUF<int>(buf);
	    new_dim.w = readBUF<int>(buf);
	    assert(buf == bufCheck + length);
	}

	~ReshapeRT(){

	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{ c,h,w};
	}

	void configureWithFormat (const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type,PluginFormat format, int maxBatchSize) NOEXCEPT override {
	}

	int initialize()  NOEXCEPT override {return 0;}

	virtual void terminate() NOEXCEPT override {}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {	return 0;}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
		return 0;
	}

	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 4*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a = buf;
		tk::dnn::writeBUF(buf, n);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
	    return true;
	    //todo assert
	}

	const char *getPluginType() const NOEXCEPT override{
	    return "1";
	}

	const char *getPluginVersion() const NOEXCEPT override{
	    return "ReshapeRT_tkDNN";
	}

	void destroy() NOEXCEPT override{delete this;}

	const char *getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}

	IPluginV2 *clone() const NOEXCEPT override{
	    ReshapeRT *p = new ReshapeRT(new_dim);
	    p->setPluginNamespace(mPluginNamespace.c_str());
	    return p;
	}

	int n, c, h, w;
	dataDim_t new_dim;
private:
    std::string mPluginNamespace;
};

class ReshapeRTPluginCreator : public IPluginCreator{
public:
    ReshapeRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("new_dim",nullptr,PluginFieldType::kUNKNOWN,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }

    IPluginV2 *deserializePlugin(const char* name,const void *serialData,size_t serialLength) NOEXCEPT override{
        ReshapeRT *pluginObj = new ReshapeRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char* name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        dataDim_t newDim = *(static_cast<const dataDim_t *>(fields[0].data));
        ReshapeRT *pluginObj = new ReshapeRT(newDim);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "ReshapeRT_tkDNN";
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

REGISTER_TENSORRT_PLUGIN(ReshapeRTPluginCreator);
