#include<cassert>
#include "../kernels.h"

class ResizeLayerRT : public IPluginV2 {

public:
	ResizeLayerRT(int c, int h, int w) {
		o_c = c;
		o_h = h;
		o_w = w;	
	}

	ResizeLayerRT(const void *data,size_t length){
	    const char *buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
	    o_c = readBUF<int>(buf);
	    o_h = readBUF<int>(buf);
	    o_w = readBUF<int>(buf);
	    i_c = readBUF<int>(buf);
	    i_h = readBUF<int>(buf);
	    i_w = readBUF<int>(buf);
	    assert(buf == bufCheck + length);
	}

	~ResizeLayerRT(){
	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{o_c, o_h, o_w};
	}

	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format,int maxBatchSize)  NOEXCEPT override {
		i_c = inputDims[0].d[0];
		i_h = inputDims[0].d[1];
		i_w = inputDims[0].d[2];        
	}

	int initialize() NOEXCEPT override {return 0;}

	virtual void terminate() NOEXCEPT override {}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {	return 0;}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {
    	// printf("%d %d %d %d %d %d\n", i_c, i_w, i_h, o_c, o_w, o_h);
        resizeForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
					  reinterpret_cast<dnnType*>(outputs[0]), 
					  batchSize, i_c, i_h, i_w, o_c, o_h, o_w, stream);
		return 0;
	}


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 6*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;

		tk::dnn::writeBUF(buf, o_c);
		tk::dnn::writeBUF(buf, o_h);
		tk::dnn::writeBUF(buf, o_w);

		tk::dnn::writeBUF(buf, i_c);
		tk::dnn::writeBUF(buf, i_h);
		tk::dnn::writeBUF(buf, i_w);
		assert(buf == a + getSerializationSize());
	}

	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
	    return true;
	    //todo assert
	}

	const char *getPluginType() const NOEXCEPT override{
	    return "ResizeLayerRT_tkDNN";
	}

	const char *getPluginVersion() const NOEXCEPT override{
	    return "1";
	}
	void destroy() NOEXCEPT override{delete this;}

	const char *getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}
	IPluginV2 *clone() const NOEXCEPT override{
	    ResizeLayerRT *p = new ResizeLayerRT(o_c,o_h,o_w);
	    p->setPluginNamespace(mPluginNamespace.c_str());
	    return p;
	}

	int i_c, i_h, i_w, o_c, o_h, o_w;

private:
    std::string mPluginNamespace;
};

class ResizeLayerRTPluginCreator : public IPluginCreator{
public:
    ResizeLayerRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("o_c",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("o_h",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("o_w",nullptr,PluginFieldType::kINT32,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }

    IPluginV2 *deserializePlugin(const char *name,const void *serialData,size_t serialLength) NOEXCEPT override{
        ResizeLayerRT *pluginObj = new ResizeLayerRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char *name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        assert(fc->nbFields == 3);
        assert(fields[0].type == PluginFieldType::kINT32);
        assert(fields[1].type == PluginFieldType::kINT32);
        assert(fields[2].type == PluginFieldType::kINT32);
        int oc = *(static_cast<const int *>(fields[0].data));
        int oh = *(static_cast<const int *>(fields[1].data));
        int ow = *(static_cast<const int *>(fields[2].data));
        ResizeLayerRT *pluginObj = new ResizeLayerRT(oc,oh,ow);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "ResizeLayerRT_tkDNN";
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

REGISTER_TENSORRT_PLUGIN(ResizeLayerRTPluginCreator);

