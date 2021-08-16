#include<cassert>

class FlattenConcatRT : public IPluginV2 {

public:
	FlattenConcatRT() {
		stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf ("CUBLAS initialization failed\n");
			return;
  		}
	}

	FlattenConcatRT(const void *data,size_t length){
	    const char *buf = reinterpret_cast<const char *>(data),*bufCheck=buf;
	    c = readBUF<int>(buf);
	    h = readBUF<int>(buf);
	    w = readBUF<int>(buf);
	    rows = readBUF<int>(buf);
	    cols = readBUF<int>(buf);
	    assert(buf == bufCheck + length);
	}

	~FlattenConcatRT(){

	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		return Dims3{ inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2], 1, 1};
	}

	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format,int maxBatchSize) NOEXCEPT override {
		assert(nbOutputs == 1 && nbInputs ==1);
		rows = inputDims[0].d[0];
		cols = inputDims[0].d[1] * inputDims[0].d[2];
		c = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
		h = 1;
		w = 1;
	}

	int initialize() NOEXCEPT override {return 0;}

	virtual void terminate() NOEXCEPT  override {	checkERROR(cublasDestroy(handle));}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {return 0;}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {
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

	virtual size_t getSerializationSize() const NOEXCEPT override {
		return 5*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a = buf;
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		tk::dnn::writeBUF(buf, rows);
		tk::dnn::writeBUF(buf, cols);
		assert(buf == a + getSerializationSize());
	}

	void destroy() NOEXCEPT override{delete this;}

	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override{
	    return true;
	}

	const char *getPluginType() const NOEXCEPT override{
	    return "FlattenConcatRT_tkDNN";
	}

	const char *getPluginVersion() const NOEXCEPT override{
	    return "1";
	}

	const char *getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}

	IPluginV2 *clone() const NOEXCEPT override {
	    FlattenConcatRT *p = new FlattenConcatRT();
	    p->setPluginNamespace(mPluginNamespace.c_str());
	    return p;
	}

	int c, h, w;
	int rows, cols;
	cublasStatus_t stat; 
	cublasHandle_t handle;
private:
    std::string mPluginNamespace;
};

class FlattenConcatRTPluginCreator : public IPluginCreator{
public:
    FlattenConcatRTPluginCreator(){
        mPluginAttributes.clear();
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
        FlattenConcatRT *pluginObj = new FlattenConcatRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char *name,const PluginFieldCollection *fc) NOEXCEPT override{
        FlattenConcatRT *pluginObj = new FlattenConcatRT();
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "FlattenConcatRT_tkDNN";
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

REGISTER_TENSORRT_PLUGIN(FlattenConcatRTPluginCreator);