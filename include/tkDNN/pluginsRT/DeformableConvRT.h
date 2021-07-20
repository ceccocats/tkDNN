#include<cassert>
#include "../kernels.h"


class DeformableConvRT : public IPlugin {



public:
	DeformableConvRT(int chunk_dim, int kh, int kw, int sh, int sw, int ph, int pw, 
					int deformableGroup, int i_n, int i_c, int i_h, int i_w, 
					int o_n, int o_c, int o_h, int o_w, 
					tk::dnn::DeformConv2d *deformable = nullptr) {
		this->chunk_dim = chunk_dim;
		this->kh = kh;
		this->kw = kw;
		this->sh = sh;
		this->sw = sw;
		this->ph = ph;
		this->pw = pw;
		this->deformableGroup = deformableGroup;
		this->i_n = i_n;
		this->i_c = i_c;
		this->i_h = i_h;
		this->i_w = i_w;
		this->o_n = o_n;
		this->o_c = o_c;
		this->o_h = o_h;
		this->o_w = o_w;
		height_ones = (i_h + 2 * ph - (1 * (kh - 1) + 1)) / sh + 1;
		width_ones = (i_w + 2 * pw - (1 * (kw - 1) + 1)) / sw + 1;
		dim_ones = i_c * kh * kw * 1 * height_ones * width_ones;
		std::cout<<i_c * o_c * kh * kw * 1<<"\n";
		checkCuda( cudaMalloc(&data_d, i_c * o_c * kh * kw * 1 * sizeof(dnnType)));
		checkCuda( cudaMalloc(&bias2_d, o_c*sizeof(dnnType)));
		checkCuda( cudaMalloc(&ones_d1, height_ones * width_ones * sizeof(dnnType)));
		checkCuda( cudaMalloc(&offset, 2*chunk_dim*sizeof(dnnType)));
		checkCuda( cudaMalloc(&mask, chunk_dim*sizeof(dnnType)));
		checkCuda( cudaMalloc(&ones_d2, dim_ones*sizeof(dnnType)));
		if(deformable != nullptr) {
			this->defRT = deformable;
			checkCuda( cudaMemcpy(data_d, deformable->data_d, sizeof(dnnType)*i_c * o_c * kh * kw * 1, cudaMemcpyDeviceToDevice) );
			checkCuda( cudaMemcpy(bias2_d, deformable->bias2_d, sizeof(dnnType)*o_c, cudaMemcpyDeviceToDevice) );
            checkCuda( cudaMemcpy(ones_d1, deformable->ones_d1, sizeof(dnnType)*height_ones*width_ones, cudaMemcpyDeviceToDevice) );
            checkCuda( cudaMemcpy(offset, deformable->offset, sizeof(dnnType)*2*chunk_dim, cudaMemcpyDeviceToDevice) );
            checkCuda( cudaMemcpy(mask, deformable->mask, sizeof(dnnType)*chunk_dim, cudaMemcpyDeviceToDevice) );
            checkCuda( cudaMemcpy(ones_d2, deformable->ones_d2, sizeof(dnnType)*dim_ones, cudaMemcpyDeviceToDevice) );
		}
		stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS)
			FatalError("CUBLAS initialization failed\n");
	}

	~DeformableConvRT() {
		checkCuda( cudaFree(data_d) );
		checkCuda( cudaFree(bias2_d) );
		checkCuda( cudaFree(ones_d1) );
		checkCuda( cudaFree(offset) );
		checkCuda( cudaFree(mask) );
		checkCuda( cudaFree(ones_d2) );
		cublasDestroy(handle);
	}

	int getNbOutputs() const override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
		return DimsCHW{defRT->output_dim.c, defRT->output_dim.h, defRT->output_dim.w};
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override { }

	int initialize() override {
		return 0;
	}

	virtual void terminate() override { }

	virtual size_t getWorkspaceSize(int maxBatchSize) const override {
		return 0;
	}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *output_conv = (dnnType*)reinterpret_cast<const dnnType*>(inputs[1]);

		// split conv2d outputs into offset to mask
		for(int b=0; b<batchSize; b++) {
			checkCuda(cudaMemcpy(offset, output_conv + b * 3 * chunk_dim, 2*chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice)); 
			checkCuda(cudaMemcpy(mask, output_conv + b * 3 * chunk_dim + 2*chunk_dim, chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice)); 
			// kernel sigmoid
			activationSIGMOIDForward(mask, mask, chunk_dim);
			// deformable convolution
			dcnV2CudaForward(stat, handle, 
								srcData, data_d,
								bias2_d, ones_d1,
								offset, mask,
								reinterpret_cast<dnnType*>(outputs[0]), ones_d2,
								kh, kw,
								sh, sw,
								ph, pw,
								1, 1,
								deformableGroup, b,
								i_n, i_c, i_h, i_w,
								o_n, o_c, o_h, o_w,
								chunk_dim);
		}
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 16 * sizeof(int) + chunk_dim * 3 * sizeof(dnnType) + (i_c * o_c * kh * kw * 1 ) * sizeof(dnnType) +
			   o_c * sizeof(dnnType) + height_ones * width_ones * sizeof(dnnType) + dim_ones * sizeof(dnnType);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, chunk_dim);
		tk::dnn::writeBUF(buf, kh);
		tk::dnn::writeBUF(buf, kw);
		tk::dnn::writeBUF(buf, sh);
		tk::dnn::writeBUF(buf, sw);
		tk::dnn::writeBUF(buf, ph);
		tk::dnn::writeBUF(buf, pw);
		tk::dnn::writeBUF(buf, deformableGroup);
		tk::dnn::writeBUF(buf, i_n);
		tk::dnn::writeBUF(buf, i_c);
		tk::dnn::writeBUF(buf, i_h);
		tk::dnn::writeBUF(buf, i_w);
		tk::dnn::writeBUF(buf, o_n);
		tk::dnn::writeBUF(buf, o_c);
		tk::dnn::writeBUF(buf, o_h);
		tk::dnn::writeBUF(buf, o_w);
		dnnType *aus = new dnnType[chunk_dim*2];
		checkCuda( cudaMemcpy(aus, offset, sizeof(dnnType)*2*chunk_dim, cudaMemcpyDeviceToHost) );
        for(int i=0; i<chunk_dim*2; i++)
    		tk::dnn::writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[chunk_dim];
		checkCuda( cudaMemcpy(aus, mask, sizeof(dnnType)*chunk_dim, cudaMemcpyDeviceToHost) );
        for(int i=0; i<chunk_dim; i++)
    		tk::dnn::writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[(i_c * o_c * kh * kw * 1 )];
		checkCuda( cudaMemcpy(aus, data_d, sizeof(dnnType)*(i_c * o_c * kh * kw * 1 ), cudaMemcpyDeviceToHost) );
        for(int i=0; i<(i_c * o_c * kh * kw * 1 ); i++)
    		tk::dnn::writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[o_c];
		checkCuda( cudaMemcpy(aus, bias2_d, sizeof(dnnType)*o_c, cudaMemcpyDeviceToHost) );
        for(int i=0; i < o_c; i++)
    		tk::dnn::writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[height_ones * width_ones];
		checkCuda( cudaMemcpy(aus, ones_d1, sizeof(dnnType)*height_ones * width_ones, cudaMemcpyDeviceToHost) );
        for(int i=0; i<height_ones * width_ones; i++)
    		tk::dnn::writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[dim_ones];
		checkCuda( cudaMemcpy(aus, ones_d2, sizeof(dnnType)*dim_ones, cudaMemcpyDeviceToHost) );
        for(int i=0; i<dim_ones; i++)
    		tk::dnn::writeBUF(buf, aus[i]);
		free(aus);
		assert(buf == a + getSerializationSize());
	}

	cublasStatus_t stat; 
	cublasHandle_t handle; 
	int i_n, i_c, i_h, i_w;
	int o_n, o_c, o_h, o_w;
	int size;
	int chunk_dim;
	int kh, kw;
	int sh, sw;
	int ph, pw;
	int deformableGroup;
	int height_ones;
	int width_ones;
	int dim_ones;
		
	dnnType *data_d;
    dnnType *bias2_d;
	dnnType *ones_d1;
	dnnType * offset;
	dnnType * mask;
	dnnType *ones_d2;
	// dnnType *input_n;
	// dnnType *offset_n;
	// dnnType *mask_n;
	// dnnType *output_n;
	
	
	tk::dnn::DeformConv2d *defRT;
};
