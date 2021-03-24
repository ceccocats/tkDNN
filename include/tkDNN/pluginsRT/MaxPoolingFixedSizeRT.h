#include<cassert>
#include "../kernels.h"

class MaxPoolFixedSizeRT : public IPlugin {

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

	~MaxPoolFixedSizeRT(){
	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{this->c, this->h, this->w};
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

		//std::cout<<this->n<<"  "<<this->c<<"  "<<this->h<<"  "<<this->w<<"  "<<this->stride_H<<"  "<<this->stride_W<<"  "<<this->winSize<<"  "<<this->padding<<std::endl;
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		MaxPoolingForward(srcData, dstData, batchSize, this->c, this->h, this->w, this->stride_H, this->stride_W, this->winSize, this->padding, stream);
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 8*sizeof(int);
	}

	virtual void serialize(void* buffer) override {
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

	int n, c, h, w;
	int stride_H, stride_W;
	int winSize;
	int padding;
};
