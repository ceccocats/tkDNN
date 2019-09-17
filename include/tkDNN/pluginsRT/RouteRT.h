#include<cassert>
#include "../kernels.h"

class RouteRT : public IPlugin {

public:
	RouteRT() {
	}

	~RouteRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		int out_c = 0;
		for(int i=0; i<nbInputDims; i++) out_c += inputs[i].d[0];
		return DimsCHW{out_c, inputs[0].d[1], inputs[0].d[2]};
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {
		in = nbInputs;
		c = 0;
		for(int i=0; i<nbInputs; i++) {
			c_in[i] = inputDims[i].d[0]; 
			c += inputDims[i].d[0];
		}
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

		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		int offset = 0;
		for(int i=0; i<in; i++) {
			dnnType *input = (dnnType*)reinterpret_cast<const dnnType*>(inputs[i]);
			int in_dim = c_in[i]*h*w;
			checkCuda( cudaMemcpyAsync(dstData + offset, input, in_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream) );
			offset += in_dim;
		}

		return 0;
	}


	virtual size_t getSerializationSize() override {
		return (4+MAX_INPUTS)*sizeof(int);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer);
		tk::dnn::writeBUF(buf, in);
		for(int i=0; i<MAX_INPUTS; i++)
			tk::dnn::writeBUF(buf, c_in[i]);

		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
	}

	static const int MAX_INPUTS = 4;
	int in;
	int c_in[MAX_INPUTS];
	int c, h, w;
};
