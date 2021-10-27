#ifndef ACTIVATION_LEAKY_RT_H
#define ACTIVATION_LEAKY_RT_H

#if NV_TENSORRT_MAJOR < 6

#include <cassert>
#include <vector>

#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"

namespace tk { namespace dnn {

class ActivationLeakyRT final : public IPlugin {

public:
	ActivationLeakyRT(float s) {
		slope = s;
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

	void terminate() override {
	}

	size_t getWorkspaceSize(int maxBatchSize) const override {
		return 0;
	}

	int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {

		activationLEAKYForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]),
											reinterpret_cast<dnnType*>(outputs[0]), batchSize*size, slope, stream);
		return 0;
	}


	size_t getSerializationSize() override {
		return 1*sizeof(int) + 1*sizeof(float);
	}

	void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		writeBUF(buf, size);
		assert(buf == a + getSerializationSize());
	}

	int size;
	float slope;
};
}}

#endif

#endif // ACTIVATION_LEAKY_RT_H