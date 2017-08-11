#include<cassert>
#include "kernels.h"

class RegionRT : public IPlugin {

public:
	RegionRT(int classes, int coords, int num, float thresh) {

		this->classes = classes;
		this->coords = coords;
		this->num = num;
		this->thresh = thresh;
	}

	~RegionRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return inputs[0];
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

		for (int b = 0; b < batchSize; ++b){
			for(int n = 0; n < num; ++n){
				int index = entry_index(b, n*w*h, 0, batchSize);
				activationLOGISTICForward(srcData + index, dstData + index, 2*w*h, stream);
				
				index = entry_index(b, n*w*h, coords, batchSize);
				activationLOGISTICForward(srcData + index, dstData + index, w*h, stream);
			}
		}

		//softmax start
		int index = entry_index(0, 0, coords + 1, batchSize);
		softmaxForward(	srcData + index, classes, batchSize*num, 
						(batchSize*c*h*w)/num, 
						w*h, 1, w*h, 1, dstData + index, stream);

		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 0;
	}

	virtual void serialize(void* buffer) override {
	}

	int c, h, w;
    int classes, coords, num;
    float thresh;

	int entry_index(int batch, int location, int entry, int batchSize) {
		int n =   location / (w*h);
		int loc = location % (w*h);
		return batch*c*h*w*batchSize + n*w*h*(coords+classes+1) + entry*w*h + loc;
	}

};
