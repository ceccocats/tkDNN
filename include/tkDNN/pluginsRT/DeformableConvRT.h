#include<cassert>
#include "../kernels.h"


class DeformableConvRT : public IPlugin {



public:
	DeformableConvRT(tk::dnn::DeformConv2d *deformable) {
		this->defRT = deformable;
	}

	~DeformableConvRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{defRT->output_dim.c, defRT->output_dim.h, defRT->output_dim.w};
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {
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
		dnnType *output_conv = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);

		// split conv2d outputs into offset to mask
		checkCuda(cudaMemcpy(defRT->offset, defRT->output_conv, 2*defRT->chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice)); 
		checkCuda(cudaMemcpy(defRT->mask, defRT->output_conv + 2*defRT->chunk_dim, defRT->chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice)); 
		// kernel sigmoide
		activationSIGMOIDForward(defRT->mask, defRT->mask, defRT->chunk_dim);
	
		// deformable convolution
		dcn_v2_cuda_forward(srcData, defRT->data_d,
							defRT->bias2_d, defRT->ones_d1,
							defRT->offset, defRT->mask,
							reinterpret_cast<dnnType*>(outputs[0]), defRT->ones_d2,
							defRT->kernelH, defRT->kernelW,
							defRT->strideH, defRT->strideW,
							defRT->paddingH, defRT->paddingW,
							1, 1,
							defRT->deformableGroup, 
							defRT->preconv->input_dim.n, defRT->preconv->input_dim.c, defRT->preconv->input_dim.h, defRT->preconv->input_dim.w,
							defRT->output_dim.n, defRT->output_dim.c, defRT->output_dim.h, defRT->output_dim.w,
							defRT->chunk_dim);

		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 0;
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer);
	}

	int size;
	tk::dnn::DeformConv2d *defRT;
};
