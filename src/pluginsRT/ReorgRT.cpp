#include<cassert>
#include "kernels.h"

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

		reorgForward((value_type*)reinterpret_cast<const value_type*>(inputs[0]), 
					  reinterpret_cast<value_type*>(outputs[0]), 
					  batchSize, c, h, w, stride);
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 0;
	}

	virtual void serialize(void* buffer) override {
	}

	int c, h, w, stride;
};
