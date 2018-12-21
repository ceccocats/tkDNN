#include<cassert>
#include "kernels.h"

class ShortcutRT : public IPlugin {

public:
	ShortcutRT(ITensor *tens) {
		this->back_layer = tens;
	}

	~ShortcutRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
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

		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 3*sizeof(int);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
	}

	int c, h, w;
	ITensor *back_layer;
};
