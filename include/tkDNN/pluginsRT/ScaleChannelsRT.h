#include<cassert>
#include "../kernels.h"

class ScaleChannelsRT : public IPlugin {

public:
	ScaleChannelsRT(tk::dnn::dataDim_t bdim, int scale_wh) {
		this->bc = bdim.c;
		this->bh = bdim.h;
		this->bw = bdim.w;
		this->scale_wh = scale_wh;
	}

	~ScaleChannelsRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{bc, bh, bw};
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
		dnnType *srcDataBack = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		int size = batchSize * bc * bh * bw;
		int channel_size = bh * bw;
		int batch_size = bc * bh * bw;
		scaleChannelsForward(srcDataBack, size, channel_size, batch_size, scale_wh, srcData, dstData, stream);

		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 7*sizeof(int);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer);
		tk::dnn::writeBUF(buf, bc);
		tk::dnn::writeBUF(buf, bh);
		tk::dnn::writeBUF(buf, bw);
		tk::dnn::writeBUF(buf, scale_wh);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
	}

	int c, h, w;
	int scale_wh;
	int bc, bh, bw;
};
