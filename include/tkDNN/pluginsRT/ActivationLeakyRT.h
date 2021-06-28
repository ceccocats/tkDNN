#include<cassert>
#include "../kernels.h"

class ActivationLeakyRT : public IPluginV2 {

public:
	ActivationLeakyRT() {


	}

	~ActivationLeakyRT(){

	}

	int getNbOutputs() const noexcept override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override {
		return inputs[0];
	}

	//void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {
	//	size = 1;
	//	for(int i=0; i<outputDims[0].nbDims; i++)
	//		size *= outputDims[0].d[i];
	//}

	int initialize() noexcept override {

		return 0;
	}

	virtual void terminate() noexcept override {
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
		return 0;
	}

	virtual int enqueue(int32_t batchSize, void const *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override {

		activationLEAKYForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
											reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, stream);
		return 0;
	}


	virtual size_t getSerializationSize() const noexcept override {
		return 1*sizeof(int);
	}

	virtual void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, size);
		assert(buf == a + getSerializationSize());
	}

	int size;
};
