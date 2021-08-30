#include<cassert>
#include "../kernels.h"


class ShortcutRT : public IPluginV2 {

public:
	ShortcutRT(tk::dnn::dataDim_t bdim, bool mul) {
	    bDim = bdim;
		this->bc = bDim.c;
		this->bh = bDim.h;
		this->bw = bDim.w;
		this->mul = mul;
	}

	~ShortcutRT(){}

	ShortcutRT(const void* data,size_t length){
	    const char* buf =reinterpret_cast<const char*>(data),*bufCheck = buf;
	    bDim.c = readBUF<int>(buf);
	    bDim.h = readBUF<int>(buf);
	    bDim.w = readBUF<int>(buf);
	    bDim.l = 1;
	    mul = readBUF<bool>(buf);
	    c = readBUF<int>(buf);
	    h = readBUF<int>(buf);
	    w = readBUF<int>(buf);
	    assert(buf == bufCheck + length);
	}

	int getNbOutputs() const NOEXCEPT override {return 1;}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
	}

	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format,int maxBatchSize) NOEXCEPT override {
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
	}

	int initialize() NOEXCEPT override {return 0;}

	virtual void terminate() NOEXCEPT override {}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {	return 0;}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {

		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *srcDataBack = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
		shortcutForward(srcDataBack, dstData, batchSize, c, h, w, 1, batchSize, bc, bh, bw, 1, mul, stream);

		return 0;
	}


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 6*sizeof(int) + sizeof(bool);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, bc);
		tk::dnn::writeBUF(buf, bh);
		tk::dnn::writeBUF(buf, bw);
		tk::dnn::writeBUF(buf, mul);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
		
	}

	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
	    return true;
	}

	const char* getPluginType() const NOEXCEPT override{
	    return "1";
	}

	const char* getPluginVersion() const NOEXCEPT override{
	    return "ShortcutRT_tkDNN";
	}

	void destroy() NOEXCEPT override{delete this;}

	const char* getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}

	IPluginV2 *clone() const NOEXCEPT override{
	    ShortcutRT *p = new ShortcutRT(bDim,mul);
	    p->setPluginNamespace(mPluginNamespace.c_str());
	    return p;
	}

	int c, h, w;
	int bc, bh, bw;
	bool mul;
	tk::dnn::dataDim_t bDim;
private:
    std::string mPluginNamespace;
};


class ShortcutRTPluginCreator : public IPluginCreator {
public:
    ShortcutRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("bDim",nullptr,PluginFieldType::kUNKNOWN,1));
        mPluginAttributes.emplace_back(PluginField("mul",nullptr,PluginFieldType::kUNKNOWN,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }

    IPluginV2 *deserializePlugin(const char *name,const void *serialData,size_t serialLength) NOEXCEPT override{
        ShortcutRT *pluginObj = new ShortcutRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char *name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        //todo assert
        tk::dnn::dataDim_t bdim = *(static_cast<const tk::dnn::dataDim_t *>(fields[0].data));
        bool mul = *(static_cast<const bool *>(fields[1].data));
        ShortcutRT *pluginObj = new ShortcutRT(bdim,mul);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "ShortcutRT_tkDNN";
    }

    const char *getPluginVersion() const NOEXCEPT override{
        return "1";
    }

    const PluginFieldCollection *getFieldNames() NOEXCEPT override{
        return &mFC;
    }
public:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};

REGISTER_TENSORRT_PLUGIN(ShortcutRTPluginCreator);