#include<cassert>
#include "../kernels.h"

class RouteRT : public IPluginV2 {

	/**
		THIS IS NOT USED ANYMORE
	*/

public:
	RouteRT(int groups, int group_id) {
		this->groups = groups;
		this->group_id = group_id;
	}

	~RouteRT(){

	}

	RouteRT(const void* data,size_t length){
	    const char* buf = reinterpret_cast<const char*>(data),*bufCheck = buf;
	    groups = readBUF<int>(buf);
	    group_id = readBUF<int>(buf);
	    in = readBUF<int>(buf);
	    for(int i=0;i <MAX_INPUTS;i++){
	        c_in[i] = readBUF<int>(buf);
	    }
	    c= readBUF<int>(buf);
	    h = readBUF<int>(buf);
	    w = readBUF<int>(buf);
	    assert(buf == bufCheck + length);
	}

	int getNbOutputs() const NOEXCEPT override {
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override {
		int out_c = 0;
		for(int i=0; i<nbInputDims; i++) out_c += inputs[i].d[0];
		return Dims3{out_c/groups, inputs[0].d[1], inputs[0].d[2]};
	}

	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,DataType type,PluginFormat format,int maxBatchSize) NOEXCEPT override {
		in = nbInputs;
		c = 0;
		for(int i=0; i<nbInputs; i++) {
			c_in[i] = inputDims[i].d[0]; 
			c += inputDims[i].d[0];
		}
		h = inputDims[0].d[1];
		w = inputDims[0].d[2];
		c /= groups;
	}

	int initialize() NOEXCEPT override {	return 0;}

	virtual void terminate() NOEXCEPT override {}

	virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override {return 0;}

	virtual int enqueue(int batchSize, const void*const * inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override {
		dnnType *dstData = reinterpret_cast<dnnType*>(outputs[0]);
		for(int b=0; b<batchSize; b++) {
			int offset = 0;
			for(int i=0; i<in; i++) {
				dnnType *input = (dnnType*)reinterpret_cast<const dnnType*>(inputs[i]);
				int in_dim = c_in[i]*h*w;
				int part_in_dim = in_dim / this->groups;
				checkCuda( cudaMemcpyAsync(dstData + b*c*w*h + offset, input + b*c*w*h*groups + this->group_id*part_in_dim, part_in_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice, stream) );
				offset += part_in_dim;
			}
		}
		return 0;
	}

	virtual size_t getSerializationSize() const NOEXCEPT override {
		return (6+MAX_INPUTS)*sizeof(int);
	}

	virtual void serialize(void* buffer) const NOEXCEPT override {
		char *buf = reinterpret_cast<char*>(buffer),*a=buf;
		tk::dnn::writeBUF(buf, groups);
		tk::dnn::writeBUF(buf, group_id);
		tk::dnn::writeBUF(buf, in);
		for(int i=0; i<MAX_INPUTS; i++)
			tk::dnn::writeBUF(buf, c_in[i]);

		tk::dnn::writeBUF(buf, c);
		tk::dnn::writeBUF(buf, h);
		tk::dnn::writeBUF(buf, w);
		assert(buf == a + getSerializationSize());
	}

	const char *getPluginType() const NOEXCEPT override{
	    return "RouteRT_tkDNN";
	}

	const char *getPluginVersion() const NOEXCEPT override{
	    return "1";
	}

	void destroy() NOEXCEPT override {delete this; }

	const char* getPluginNamespace() const NOEXCEPT override{
	    return mPluginNamespace.c_str();
	}

	void setPluginNamespace(const char *pluginNamespace) NOEXCEPT override{
	    mPluginNamespace = pluginNamespace;
	}

	bool supportsFormat(DataType type,PluginFormat format) const NOEXCEPT override { return true;}

	IPluginV2* clone() const NOEXCEPT override{
	    RouteRT *p = new RouteRT(groups,group_id);
	    p->setPluginNamespace(mPluginNamespace.c_str());
	    return p;
	}
	static const int MAX_INPUTS = 4;
	int in;
	int c_in[MAX_INPUTS];
	int c, h, w;
	int groups, group_id;
private:
    std::string mPluginNamespace;
};

class RouteRTPluginCreator : public IPluginCreator{
public:
    RouteRTPluginCreator(){
        mPluginAttributes.emplace_back(PluginField("groups",nullptr,PluginFieldType::kINT32,1));
        mPluginAttributes.emplace_back(PluginField("group_id",nullptr,PluginFieldType::kINT32,1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override{
        mPluginNamespace = pluginNamespace;
    }

    const char *getPluginNamespace() const NOEXCEPT override{
        return mPluginNamespace.c_str();
    }

    IPluginV2 *deserializePlugin(const char* name,const void* serialData,size_t serialLength) NOEXCEPT override{
        RouteRT *pluginObj = new RouteRT(serialData,serialLength);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    IPluginV2 *createPlugin(const char* name,const PluginFieldCollection *fc) NOEXCEPT override{
        const PluginField *fields = fc->fields;
        assert(fc->nbFields == 2);
        assert(fields[0].type == PluginFieldType::kINT32);
        assert(fields[1].type == PluginFieldType::kINT32);
        int groups = *(static_cast<const int *>(fields[0].data));
        int group_id = *(static_cast<const int *>(fields[1].data));
        RouteRT *pluginObj = new RouteRT(groups,group_id);
        pluginObj->setPluginNamespace(mPluginNamespace.c_str());
        return pluginObj;
    }

    const char *getPluginName() const NOEXCEPT override{
        return "RouteRT_tkDNN";
    }

    const char *getPluginVersion() const NOEXCEPT override{
        return "1";
    }

    const PluginFieldCollection *getFieldNames() NOEXCEPT override{
        return &mFC;
    }
 private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};

REGISTER_TENSORRT_PLUGIN(RouteRTPluginCreator);
