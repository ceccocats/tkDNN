#include<cassert>
#include "../kernels.h"

class ActivationReLUCeiling : public IPlugin {

public:
	ActivationReLUCeiling(const float ceiling) {
		this->ceiling = ceiling;
	}

	~ActivationReLUCeiling(){

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

		activationReLUCeilingForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
											reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, ceiling, stream);
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 1*sizeof(int) +  1*sizeof(float);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, ceiling);
		tk::dnn::writeBUF(buf, size);
		assert(buf = a + getSerializationSize());
		
	}

	int size;
	float ceiling;
};
