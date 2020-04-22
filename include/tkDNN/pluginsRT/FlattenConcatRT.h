#include<cassert>

class FlattenConcatRT : public IPlugin {

public:
	FlattenConcatRT() {
		stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf ("CUBLAS initialization failed\n");
			return;
  		}
	}

	~FlattenConcatRT(){

	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{ inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2], 1, 1};
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {
		assert(nbOutputs == 1 && nbInputs ==1);
		rows = inputDims[0].d[0];
		cols = inputDims[0].d[1] * inputDims[0].d[2];
		c = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
		h = 1;
		w = 1;
	}

	int initialize() override {
		return 0;
	}

	virtual void terminate() override {
		checkERROR(cublasDestroy(handle));
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override {
		return 0;
	}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*rows*cols*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));

		checkERROR( cublasSetStream(handle, stream) );	
		for(int i=0; i<batchSize; i++) {
			float const alpha(1.0);
			float const beta(0.0);
			int offset = i*rows*cols;
			checkERROR( cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, srcData + offset, cols, &beta, srcData + offset, rows, dstData + offset, rows ));
		}
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 5*sizeof(int);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		tk::dnn::writeBUF(buf, rows);
		tk::dnn::writeBUF(buf, cols);
	}

	int c, h, w;
	int rows, cols;
	cublasStatus_t stat; 
	cublasHandle_t handle; 
};
