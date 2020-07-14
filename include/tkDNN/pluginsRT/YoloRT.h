#include<cassert>
#include "../kernels.h"

#define YOLORT_CLASSNAME_W 256

class YoloRT : public IPlugin {



public:
	YoloRT(int classes, int num, tk::dnn::Yolo *yolo = nullptr, int n_masks=3, float scale_xy=1) {

		this->classes = classes;
		this->num = num;
		this->n_masks = n_masks;
		this->scaleXY = scale_xy;

        mask = new dnnType[n_masks];
        bias = new dnnType[num*n_masks*2];
        if(yolo != nullptr) {
            memcpy(mask, yolo->mask_h, sizeof(dnnType)*n_masks);
            memcpy(bias, yolo->bias_h, sizeof(dnnType)*num*n_masks*2);
			classesNames = yolo->classesNames;
        }
	}

	~YoloRT(){

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
			for(int n = 0; n < n_masks; ++n){
				int index = entry_index(b, n*w*h, 0);
				activationLOGISTICForward(srcData + index, dstData + index, 2*w*h, stream);

				if (this->scaleXY != 1) scalAdd(dstData + index, 2 * w*h, this->scaleXY, -0.5*(this->scaleXY - 1), 1);
				
				index = entry_index(b, n*w*h, 4);
				activationLOGISTICForward(srcData + index, dstData + index, (1+classes)*w*h, stream);
			}
		}

		//std::cout<<"YOLO END\n";
		return 0;
	}


	virtual size_t getSerializationSize() override {
		return 6*sizeof(int) + sizeof(float)+ n_masks*sizeof(dnnType) + num*n_masks*2*sizeof(dnnType) + YOLORT_CLASSNAME_W*classes*sizeof(char);
	}

	virtual void serialize(void* buffer) override {
		char *buf = reinterpret_cast<char*>(buffer);
		tk::dnn::writeBUF(buf, classes);
		tk::dnn::writeBUF(buf, num);
		tk::dnn::writeBUF(buf, n_masks);
		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		tk::dnn::writeBUF(buf, scaleXY);
        for(int i=0; i<n_masks; i++)
    		tk::dnn::writeBUF(buf, mask[i]);
        for(int i=0; i<n_masks*2*num; i++)
    		tk::dnn::writeBUF(buf, bias[i]);

		// save classes names
		for(int i=0; i<classes; i++) {
			char tmp[YOLORT_CLASSNAME_W];
			strcpy(tmp, classesNames[i].c_str());
			for(int j=0; j<YOLORT_CLASSNAME_W; j++) {
				tk::dnn::writeBUF(buf, tmp[j]);
			}
		}
	}

	int c, h, w;
    int classes, num, n_masks;
	float scaleXY;
	std::vector<std::string> classesNames;

    dnnType *mask;
    dnnType *bias;

	int entry_index(int batch, int location, int entry) {
		int n =   location / (w*h);
		int loc = location % (w*h);
		return batch*c*h*w + n*w*h*(4+classes+1) + entry*w*h + loc;
	}

};
