#ifndef DEFORMABLE_CONV_RT_H
#define DEFORMABLE_CONV_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../kernels.h"
#include "../buffer_func.h"
#include "../Layer.h"

#define PLUGIN_NAME "Deformable"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class DeformableConvRT final : public nvinfer1::IPluginV2 {

public:
	DeformableConvRT(int chunk_dim, int kh, int kw, int sh, int sw, int ph, int pw,
					int deformableGroup, int i_n, int i_c, int i_h, int i_w,
					int o_n, int o_c, int o_h, int o_w,
					DeformConv2d *deformable = nullptr) {
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

	int getNbOutputs() const noexcept override {
		return 1;
	}

	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override {
		return nvinfer1::Dims3{defRT->output_dim.c, defRT->output_dim.h, defRT->output_dim.w};
	}

	void configureWithFormat(nvinfer1::Dims const * inputDims,
					int32_t nbInputs,
					nvinfer1::Dims const * outputDims,
					int32_t nbOutputs,
					nvinfer1::DataType type,
					nvinfer1::PluginFormat format,
					int32_t maxBatchSize) noexcept override {
	}

	int initialize() noexcept override {
		return 0;
	}

	void terminate() noexcept override { }

	size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
		return 0;
	}

	int32_t enqueue(int32_t batchSize, const void* const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
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

	size_t getSerializationSize() const noexcept override {
		return 16 * sizeof(int) + chunk_dim * 3 * sizeof(dnnType) + (i_c * o_c * kh * kw * 1 ) * sizeof(dnnType) +
			   o_c * sizeof(dnnType) + height_ones * width_ones * sizeof(dnnType) + dim_ones * sizeof(dnnType);
	}

	void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		writeBUF(buf, chunk_dim);
		writeBUF(buf, kh);
		writeBUF(buf, kw);
		writeBUF(buf, sh);
		writeBUF(buf, sw);
		writeBUF(buf, ph);
		writeBUF(buf, pw);
		writeBUF(buf, deformableGroup);
		writeBUF(buf, i_n);
		writeBUF(buf, i_c);
		writeBUF(buf, i_h);
		writeBUF(buf, i_w);
		writeBUF(buf, o_n);
		writeBUF(buf, o_c);
		writeBUF(buf, o_h);
		writeBUF(buf, o_w);
		dnnType *aus = new dnnType[chunk_dim*2];
		checkCuda( cudaMemcpy(aus, offset, sizeof(dnnType)*2*chunk_dim, cudaMemcpyDeviceToHost) );
        for(int i=0; i<chunk_dim*2; i++)
    		writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[chunk_dim];
		checkCuda( cudaMemcpy(aus, mask, sizeof(dnnType)*chunk_dim, cudaMemcpyDeviceToHost) );
        for(int i=0; i<chunk_dim; i++)
    		writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[(i_c * o_c * kh * kw * 1 )];
		checkCuda( cudaMemcpy(aus, data_d, sizeof(dnnType)*(i_c * o_c * kh * kw * 1 ), cudaMemcpyDeviceToHost) );
        for(int i=0; i<(i_c * o_c * kh * kw * 1 ); i++)
    		writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[o_c];
		checkCuda( cudaMemcpy(aus, bias2_d, sizeof(dnnType)*o_c, cudaMemcpyDeviceToHost) );
        for(int i=0; i < o_c; i++)
    		writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[height_ones * width_ones];
		checkCuda( cudaMemcpy(aus, ones_d1, sizeof(dnnType)*height_ones * width_ones, cudaMemcpyDeviceToHost) );
        for(int i=0; i<height_ones * width_ones; i++)
    		writeBUF(buf, aus[i]);
		free(aus);
		aus = new dnnType[dim_ones];
		checkCuda( cudaMemcpy(aus, ones_d2, sizeof(dnnType)*dim_ones, cudaMemcpyDeviceToHost) );
        for(int i=0; i<dim_ones; i++)
    		writeBUF(buf, aus[i]);
		free(aus);
		assert(buf == a + getSerializationSize());
	}

	// Extra IPluginV2 overrides
	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
			return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kLINEAR);
	}

	nvinfer1::IPluginV2 * clone() const noexcept override {
			auto a = new DeformableConvRT(*this);
			return a;
	}

	const char* getPluginType() const noexcept override {
			return PLUGIN_NAME;
	}

	const char* getPluginVersion() const noexcept override {
			return PLUGIN_VERSION;
	}

	void destroy() noexcept override {}

	void setPluginNamespace(const char* pluginNamespace) noexcept override {
			mNamespace = pluginNamespace;
	}

	const char* getPluginNamespace() const noexcept override {
			return mNamespace.c_str();
	}

	std::string mNamespace;

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

	DeformConv2d *defRT;
};

class DeformableConvRTCreator final : public nvinfer1::IPluginCreator {
public:
    DeformableConvRTCreator() = default;

    const char* getPluginName() const noexcept override {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override {
        return PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
        return &mFC;
    }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override {
        std::cout << "Create plugin" << std::endl;
        return nullptr;
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
			const char * buf = reinterpret_cast<const char*>(serialData),*bufCheck = buf;
			int chuck_dimTemp = readBUF<int>(buf);
			int khTemp = readBUF<int>(buf);
			int kwTemp = readBUF<int>(buf);
			int shTemp = readBUF<int>(buf);
			int swTemp = readBUF<int>(buf);
			int phTemp = readBUF<int>(buf);
			int pwTemp = readBUF<int>(buf);
			int deformableGroupTemp = readBUF<int>(buf);
			int i_nTemp = readBUF<int>(buf);
			int i_cTemp = readBUF<int>(buf);
			int i_hTemp = readBUF<int>(buf);
			int i_wTemp = readBUF<int>(buf);
			int o_nTemp = readBUF<int>(buf);
			int o_cTemp = readBUF<int>(buf);
			int o_hTemp = readBUF<int>(buf);
			int o_wTemp = readBUF<int>(buf);

			DeformableConvRT* r = new DeformableConvRT(chuck_dimTemp, khTemp, kwTemp, shTemp, swTemp, phTemp, pwTemp, deformableGroupTemp, i_nTemp, i_cTemp, i_hTemp, i_wTemp, o_nTemp, o_cTemp, o_hTemp, o_wTemp, nullptr);
			dnnType *aus = new dnnType[r->chunk_dim*2];
			for(int i=0; i<r->chunk_dim*2; i++)
			aus[i] = readBUF<dnnType>(buf);
			checkCuda( cudaMemcpy(r->offset, aus, sizeof(dnnType)*2*r->chunk_dim, cudaMemcpyHostToDevice) );
					free(aus);
			aus = new dnnType[r->chunk_dim];
			for(int i=0; i<r->chunk_dim; i++)
							aus[i] = readBUF<dnnType>(buf);
			checkCuda( cudaMemcpy(r->mask, aus, sizeof(dnnType)*r->chunk_dim, cudaMemcpyHostToDevice) );
					free(aus);
			aus = new dnnType[(r->i_c * r->o_c * r->kh * r->kw * 1 )];
			for(int i=0; i<(r->i_c * r->o_c * r->kh * r->kw * 1 ); i++)
					aus[i] = readBUF<dnnType>(buf);
			checkCuda( cudaMemcpy(r->data_d, aus, sizeof(dnnType)*(r->i_c * r->o_c * r->kh * r->kw * 1 ), cudaMemcpyHostToDevice) );
					free(aus);
			aus = new dnnType[r->o_c];
			for(int i=0; i < r->o_c; i++)
					aus[i] = readBUF<dnnType>(buf);
			checkCuda( cudaMemcpy(r->bias2_d, aus, sizeof(dnnType)*r->o_c, cudaMemcpyHostToDevice) );
					free(aus);
			aus = new dnnType[r->height_ones * r->width_ones];
			for(int i=0; i<r->height_ones * r->width_ones; i++)
					aus[i] = readBUF<dnnType>(buf);
			checkCuda( cudaMemcpy(r->ones_d1, aus, sizeof(dnnType)*r->height_ones * r->width_ones, cudaMemcpyHostToDevice) );
					free(aus);
			aus = new dnnType[r->dim_ones];
			for(int i=0; i<r->dim_ones; i++)
					aus[i] = readBUF<dnnType>(buf);
			checkCuda( cudaMemcpy(r->ones_d2, aus, sizeof(dnnType)*r->dim_ones, cudaMemcpyHostToDevice) );
					free(aus);
					assert(buf == bufCheck + serialLength);
			return r;
		}

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
			mNamespace = pluginNamespace;
		}

    const char* getPluginNamespace() const noexcept override {
			return mNamespace.c_str();
		}

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};
}}
#undef PLUGIN_NAME
#undef PLUGIN_VERSION

#endif // DEFORMABLE_CONV_RT_H