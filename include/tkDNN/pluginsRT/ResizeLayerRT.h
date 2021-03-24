#include<cassert>
#include "../kernels.h"

class ResizeLayerRT : public IPlugin {

public:
	ResizeLayerRT(int c, int h, int w) {
		o_c = c;
		o_h = h;
		o_w = w;	
	}

	~ResizeLayerRT(){
	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{o_c, o_h, o_w};
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {
		i_c = inputDims[0].d[0];
		i_h = inputDims[0].d[1];
		i_w = inputDims[0].d[2];        
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
    	// printf("%d %d %d %d %d %d\n", i_c, i_w, i_h, o_c, o_w, o_h);
        resizeForward((dnnType*)reinterpret_cast<const dnnType*>(inputs[0]), 
					  reinterpret_cast<dnnType*>(outputs[0]), 
					  batchSize, i_c, i_h, i_w, o_c, o_h, o_w, stream);
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 6*sizeof(int);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;

		tk::dnn::writeBUF(buf, o_c);
		tk::dnn::writeBUF(buf, o_h);
		tk::dnn::writeBUF(buf, o_w);

		tk::dnn::writeBUF(buf, i_c);
		tk::dnn::writeBUF(buf, i_h);
		tk::dnn::writeBUF(buf, i_w);
		assert(buf == a + getSerializationSize());
	}

	int i_c, i_h, i_w, o_c, o_h, o_w;
};
