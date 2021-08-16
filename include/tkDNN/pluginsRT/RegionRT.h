#include<cassert>
#include "../kernels.h"

class RegionRT : public IPluginV2 {

public:
	RegionRT(int classes, int coords, int num) {
		this->classes = classes;
		this->coords = coords;
		this->num = num;
	}

	~RegionRT(){

	}

	RegionRT(const void *data,size_t length){
	    const char *buf = reinterpret_cast<const char*>(data),*bufCheck=buf;
	    classes = readBUF<int>(buf);
	    coords = readBUF<int>(buf);
	    num = readBUF<int>(buf);
	    c = readBUF<int>(buf);
	    h = readBUF<int>(buf);
	    w = readBUF<int>(buf);
	    assert(buf == bufCheck+length);
	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return inputs[0];
	}

	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format, int maxBatchSize) NOEXCEPT override {
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
	}

	int initialize() NOEXCEPT override { return 0; }


	virtual void terminate() NOEXCEPT override { }

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override { return 0; }

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {

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


	virtual size_t getSerializationSize()  const NOEXCEPT override {
		return 6*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, classes);
		tk::dnn::writeBUF(buf, coords);
		tk::dnn::writeBUF(buf, num);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

	const char *getPluginType() const NOEXCEPT override{
	    return "RegionRT_tkDNN";
	}

	const char *getPluginVersion() const NOEXCEPT override{
	    return "1";
	}

	void destroy() NOEXCEPT override {delete this;}

	const char* getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}

	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
	    return true;
	}

	IPluginV2* clone() const NOEXCEPT override{
	    RegionRT *p = new RegionRT(classes,coords,num);
	    p->setPluginNamespace(mPluginNamespace.c_str());
	    return p;
	}

	int c, h, w;
    int classes, coords, num;

	int entry_index(int batch, int location, int entry) {
		int n =   location / (w*h);
		int loc = location % (w*h);
		return batch*c*h*w + n*w*h*(coords+classes+1) + entry*w*h + loc;
	}

private:
    std::string mPluginNamespace;
};

class RegionRTPluginCreator : public IPluginCreator{
public:
    RegionRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("classes",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("coords",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("num",nullptr,PluginFieldType::kINT32,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }
    IPluginV2 *deserializePlugin(const char* name,const void *serialData,size_t serialLength) NOEXCEPT override{
        RegionRT *pluginObj = new RegionRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char* name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        assert(fc->nbFields == 3);
        assert(fields[0].type == PluginFieldType::kINT32);
        assert(fields[1].type == PluginFieldType::kINT32);
        assert(fields[2].type == PluginFieldType::kINT32);
        int classes = *(static_cast<const int*>(fields[0].data));
        int coords = *(static_cast<const int*>(fields[1].data));
        int num = *(static_cast<const int*>(fields[2].data));
        RegionRT *pluginObj = new RegionRT(classes,coords,num);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;

    }

    const char *getPluginName() const NOEXCEPT override{
        return "RegionRT_tkDNN";
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

REGISTER_TENSORRT_PLUGIN(RegionRTPluginCreator);

