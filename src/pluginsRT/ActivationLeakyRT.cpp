#include<cassert>
#include "kernels.h"

class ActivationLeakyRT : public IPlugin {

public:
	ActivationLeakyRT() {


	}

	~ActivationLeakyRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return inputs[0];
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {
		size = 1;
		for(int i=0; i<outputDims[0].nbDims; i++)
			size *= outputDims[0].d[i];
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

		activationLEAKYForward((value_type*)reinterpret_cast<const value_type*>(inputs[0]), 
											reinterpret_cast<value_type*>(outputs[0]), size);
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 0;
	}

	virtual void serialize(void* buffer) override {
	}

	int size;
};
