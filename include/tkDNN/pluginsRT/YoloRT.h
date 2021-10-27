#ifndef YOLO_RT_H
#define YOLO_RT_H

#include <cassert>
#include <vector>

#include <NvInferRuntimeCommon.h>
#include <NvInfer.h>

#include "../yoloContainer.h"
#include "../kernels.h"
#include "../buffer_func.h"
#include "../Layer.h"

#define YOLORT_CLASSNAME_W 256

#define PLUGIN_NAME "Yolo"
#define PLUGIN_VERSION "1"
namespace tk { namespace dnn {

class YoloRT final : public nvinfer1::IPluginV2 {
public:
	YoloRT(int classes, int num, Yolo *yolo = nullptr, int n_masks=3, float scale_xy=1, float nms_thresh=0.45, int nms_kind=0, int new_coords=0) {
		this->classes = classes;
		this->num = num;
		this->n_masks = n_masks;
		this->scaleXY = scale_xy;
		this->nms_thresh = nms_thresh;
		this->nms_kind = nms_kind;
		this->new_coords = new_coords;

        mask = new dnnType[n_masks];
        bias = new dnnType[num*n_masks*2];
        if(yolo != nullptr) {
            memcpy(mask, yolo->mask_h, sizeof(dnnType)*n_masks);
            memcpy(bias, yolo->bias_h, sizeof(dnnType)*num*n_masks*2);
			classesNames = yolo->classesNames;
        }
	}

	~YoloRT() = default;

	int getNbOutputs() const noexcept override {
		return 1;
	}

	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) noexcept override {
		return inputs[0];
	}

	void configureWithFormat(nvinfer1::Dims const * inputDims,
					int32_t nbInputs,
					nvinfer1::Dims const * outputDims,
					int32_t nbOutputs,
					nvinfer1::DataType type,
					nvinfer1::PluginFormat format,
					int32_t maxBatchSize) noexcept override {
		c = inputDims[0].d[0];
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
	}

	int initialize() noexcept override {
		return 0;
	}

	void terminate() noexcept override {
	}

	size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
		return 0;
	}

	int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
		dnnType *srcData = (dnnType*)reinterpret_cast<const dnnType*>(inputs[0]);
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);

		checkCuda( cudaMemcpyAsync(dstData, srcData, batchSize*c*h*w*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream));


        for (int b = 0; b < batchSize; ++b){
            for(int n = 0; n < n_masks; ++n){
                int index = entry_index(b, n*w*h, 0);
                if (new_coords == 1){
                    if (this->scaleXY != 1) scalAdd(dstData + index, 2 * w*h, this->scaleXY, -0.5*(this->scaleXY - 1), 1);
                }
                else{
                    activationLOGISTICForward(srcData + index, dstData + index, 2*w*h, stream); //x,y

                    if (this->scaleXY != 1) scalAdd(dstData + index, 2 * w*h, this->scaleXY, -0.5*(this->scaleXY - 1), 1);

                    index = entry_index(b, n*w*h, 4);
                    activationLOGISTICForward(srcData + index, dstData + index, (1+classes)*w*h, stream);
                }
            }
        }

		//std::cout<<"YOLO END\n";
		return 0;
	}

	size_t getSerializationSize() const noexcept override {
		return 8*sizeof(int) + 2*sizeof(float)+ n_masks*sizeof(dnnType) + num*n_masks*2*sizeof(dnnType) + YOLORT_CLASSNAME_W*classes*sizeof(char);
	}

	void serialize(void* buffer) const noexcept override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		writeBUF(buf, classes); 	//std::cout << "Classes :" << classes << std::endl;
		writeBUF(buf, num); 		//std::cout << "Num : " << num << std::endl;
		writeBUF(buf, n_masks); 	//std::cout << "N_Masks" << n_masks << std::endl;
		writeBUF(buf, scaleXY); 	//std::cout << "ScaleXY :" << scaleXY << std::endl;
		writeBUF(buf, nms_thresh); //std::cout << "nms_thresh :" << nms_thresh << std::endl;
		writeBUF(buf, nms_kind); 	//std::cout << "nms_kind : " << nms_kind << std::endl;
		writeBUF(buf, new_coords); //std::cout << "new_coords : " << new_coords << std::endl;
		writeBUF(buf, c); 			//std::cout << "C : " << c << std::endl;
		writeBUF(buf, h); 			//std::cout << "H : " << h << std::endl;
		writeBUF(buf, w); 			//std::cout << "C : " << c << std::endl;
		for (int i = 0; i < n_masks; i++)
		{
			writeBUF(buf, mask[i]); //std::cout << "mask[i] : " << mask[i] << std::endl;
		}
		for (int i = 0; i < n_masks * 2 * num; i++)
		{
			writeBUF(buf, bias[i]); //std::cout << "bias[i] : " << bias[i] << std::endl;
		}

		// save classes names
		for(int i=0; i<classes; i++) {
			char tmp[YOLORT_CLASSNAME_W];
			strcpy(tmp, classesNames[i].c_str());
			for(int j=0; j<YOLORT_CLASSNAME_W; j++) {
				writeBUF(buf, tmp[j]);
			}
		}
		assert(buf == a + getSerializationSize());
	}

	// Extra IPluginV2 overrides
	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const noexcept override {
			return true;
	}

	nvinfer1::IPluginV2 * clone() const noexcept override {
			auto a = new YoloRT(*this);
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

	int c, h, w;
    int classes, num, n_masks;
	float scaleXY;
	float nms_thresh;
	int nms_kind;
	int new_coords;
	std::vector<std::string> classesNames;

    dnnType *mask;
    dnnType *bias;

	int entry_index(int batch, int location, int entry) {
		int n =   location / (w*h);
		int loc = location % (w*h);
		return batch*c*h*w + n*w*h*(4+classes+1) + entry*w*h + loc;
	}
};

class YoloRTCreator final : public nvinfer1::IPluginCreator {
public:
    YoloRTCreator() = default;

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

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

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

#endif // YOLO_RT_H