#include<cassert>
#include "../kernels.h"

class ReorgRT : public IPlugin {

public:
	ReorgRT(int stride) {
		this->stride = stride;
	}

	~ReorgRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{inputs[0].d[0]*stride*stride, inputs[0].d[1]/stride, inputs[0].d[2]/stride};
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
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

		reorgForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
					  reinterpret_cast<dnnType*>(outputs[0]), 
					  batchSize, c, h, w, stride, stream);
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 4*sizeof(int);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, stride);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

	int c, h, w, stride;
};
