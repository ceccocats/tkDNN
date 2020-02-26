#include<cassert>

class SoftmaxRT : public IPlugin {

public:
	SoftmaxRT(const tk::dnn::dataDim_t* dim) {
		assert(dim != nullptr);
		this->dim.n = dim->n;
		this->dim.c = dim->c;
		this->dim.h = dim->h;
		this->dim.w = dim->w;
		this->dim.l = dim->l;
	}

	~SoftmaxRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsNCHW{this->dim.n,this->dim.c,this->dim.h,this->dim.w };
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {		
	}

	int initialize() override {
		return 0;
	}

	virtual void terminate() override {
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override {
		return 0;
	}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		
		
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 5*sizeof(int);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer);
		tk::dnn::writeBUF(buf, this->dim.n);
		tk::dnn::writeBUF(buf, this->dim.c);
		tk::dnn::writeBUF(buf, this->dim.h);
		tk::dnn::writeBUF(buf, this->dim.w);
		tk::dnn::writeBUF(buf, this->dim.l);
	}

	dataDim_t dim;
};
