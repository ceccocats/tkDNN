#include<cassert>
#include "../kernels.h"


class MaxPoolFixedSizeRT : public IPluginV2 {

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

	MaxPoolFixedSizeRT(const void *data,size_t length){
	    const char *buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
	    c = readBUF<int>(buf);
	    h = readBUF<int>(buf);
	    w = readBUF<int>(buf);
	    n = readBUF<int>(buf);
	    stride_H = readBUF<int>(buf);
	    stride_W = readBUF<int>(buf);
	    winSize  = readBUF<int>(buf);
	    padding = readBUF<int>(buf);
	    assert(buf == bufCheck + length);
	}

	~MaxPoolFixedSizeRT(){
	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{this->c, this->h, this->w};
	}

	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format,int maxBatchSize) NOEXCEPT override {
	}

	int initialize() NOEXCEPT override {return 0;}

	virtual void terminate() NOEXCEPT override {}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {	return 0;}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {

		//std::cout<<this->n<<"  "<<this->c<<"  "<<this->h<<"  "<<this->w<<"  "<<this->stride_H<<"  "<<this->stride_W<<"  "<<this->winSize<<"  "<<this->padding<<std::endl;
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		MaxPoolingForward(srcData, dstData, batchSize, this->c, this->h, this->w, this->stride_H, this->stride_W, this->winSize, this->padding, stream);
		return 0;
	}


	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 8*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;

		tk::dnn::writeBUF(buf, this->c);
		tk::dnn::writeBUF(buf, this->h);
		tk::dnn::writeBUF(buf, this->w);
		tk::dnn::writeBUF(buf, this->n);
		tk::dnn::writeBUF(buf, this->stride_H);
		tk::dnn::writeBUF(buf, this->stride_W);
		tk::dnn::writeBUF(buf, this->winSize);
		tk::dnn::writeBUF(buf, this->padding);
		assert(buf == a + getSerializationSize());
	}

	void destroy() NOEXCEPT override{delete this;}

	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
	    return true;
	    //todo assert
	}

	const char *getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}
	const char *getPluginType() const NOEXCEPT override{
	    return "MaxPoolingFixedSizeRT_tkDNN";
	}

	const char *getPluginVersion() const NOEXCEPT override{
	    return "1";
	}

	IPluginV2 *clone() const NOEXCEPT override{
	    MaxPoolFixedSizeRT *p = new MaxPoolFixedSizeRT(c,h,w,n,stride_H,stride_W,winSize,padding);
	    p->setPluginNamespace(mPluginNamespace.c_str());
	    return p;
	}


	int n, c, h, w;
	int stride_H, stride_W;
	int winSize;
	int padding;

private:
    std::string mPluginNamespace;
};

class MaxPoolFixedSizeRTPluginCreator : public IPluginCreator{
public:
    MaxPoolFixedSizeRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("c",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("h",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("w",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("n",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("stride_H",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("stride_W",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("winSize",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("padding",nullptr,PluginFieldType::kINT32,1));
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
        MaxPoolFixedSizeRT *pluginObj = new MaxPoolFixedSizeRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char *name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        //todo assert
        int c = *(static_cast<const int *>(fields[0].data));
        int h = *(static_cast<const int *>(fields[1].data));
        int w = *(static_cast<const int *>(fields[2].data));
        int n = *(static_cast<const int *>(fields[3].data));
        int stride_H = *(static_cast<const int *>(fields[4].data));
        int stride_W = *(static_cast<const int *>(fields[5].data));
        int winSize = *(static_cast<const int *>(fields[6].data));
        int padding = *(static_cast<const int *>(fields[7].data));
        MaxPoolFixedSizeRT *pluginObj = new MaxPoolFixedSizeRT(c,h,w,n,stride_H,stride_W,winSize,padding);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "MaxPoolingFixedSizeRT_tkDNN";
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

REGISTER_TENSORRT_PLUGIN(MaxPoolFixedSizeRTPluginCreator);
