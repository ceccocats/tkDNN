#ifndef LAYER_H
#define LAYER_H

#include<iostream>
#include "utils.h"
#include "Network.h"

namespace tkDNN {

enum layerType_t {
    LAYER_DENSE,
    LAYER_CONV2D,
    LAYER_ACTIVATION,
    LAYER_FLATTEN,
    LAYER_MULADD,
    LAYER_POOLING,
    LAYER_SOFTMAX,
    LAYER_ROUTE,
    LAYER_REORG,
    LAYER_REGION
};

/** 
    Simple layer Father class
*/
class Layer {

public:
    Layer(Network *net);
    virtual ~Layer();
    virtual layerType_t getLayerType() = 0;

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData) {
        std::cout<<"No infer action for this layer\n";
        return NULL;
    }

    dataDim_t input_dim, output_dim;
    dnnType *dstData;  //where results will be putted

    std::string getLayerName() {
        layerType_t type = getLayerType();
        switch(type) {
            case LAYER_DENSE:       return "Dense";
            case LAYER_CONV2D:      return "Conv2d";
            case LAYER_ACTIVATION:  return "Activation";
            case LAYER_FLATTEN:     return "Flatten";
            case LAYER_MULADD:      return "MulAdd";
            case LAYER_POOLING:     return "Pooling";
            case LAYER_SOFTMAX:     return "Softmax";
            case LAYER_ROUTE:       return "Route";            
            case LAYER_REORG:       return "Reorg";
            case LAYER_REGION:      return "Region";
            default:                return "unknown";
        }
    }

protected:
    Network *net;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;

};


/**
    Father class of all layer that need to load trained weights
*/
class LayerWgs : public Layer {

public:
    LayerWgs(Network *net, int inputs, int outputs, int kh, int kw, int kt, 
             const char* fname_weights, bool batchnorm = false); 
    virtual ~LayerWgs();

    int inputs, outputs;
    std::string weights_path;

    dnnType *data_h, *data_d;
    dnnType *bias_h, *bias_d;

    //batchnorm
    bool batchnorm;
    dnnType *scales_h,   *scales_d;
    dnnType *mean_h,     *mean_d;
    dnnType *variance_h, *variance_d;
};


/**
    Dense (full interconnection) layer
*/
class Dense : public LayerWgs {

public:
    Dense(Network *net, int out_ch, const char* fname_weights); 
    virtual ~Dense();
    virtual layerType_t getLayerType() { return LAYER_DENSE; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
};


/**
    Avaible activation functions
*/
typedef enum {
    ACTIVATION_ELU     = 100,
    ACTIVATION_LEAKY   = 101
} tkdnnActivationMode_t;

/**
    Activation layer (it doesnt need weigths)
*/
class Activation : public Layer {

public:
    int act_mode;

    Activation(Network *net, int act_mode); 
    virtual ~Activation();
    virtual layerType_t getLayerType() { return LAYER_ACTIVATION; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

protected:
    cudnnActivationDescriptor_t activDesc;
};


/**
    Convolutional 2D layer
*/
class Conv2d : public LayerWgs {

public:
    Conv2d( Network *net, int out_ch, int kernelH, int kernelW, 
                int strideH, int strideW, int paddingH, int paddingW,
                const char* fname_weights, bool batchnorm = false); 
    virtual ~Conv2d();
    virtual layerType_t getLayerType() { return LAYER_CONV2D; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

    int kernelH, kernelW, strideH, strideW, paddingH, paddingW;

protected:
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnTensorDescriptor_t biasTensorDesc;

    void*  workSpace;
    size_t ws_sizeInBytes;
};


/**
    Flatten layer
    is actually a matrix transposition
*/
class Flatten : public Layer {

public:
    Flatten(Network *net); 
    virtual ~Flatten();
    virtual layerType_t getLayerType() { return LAYER_FLATTEN; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
};


/**
    MulAdd layer
    apply a multiplication and then an addition for each data
*/
class MulAdd : public Layer {

public:
    MulAdd(Network *net, dnnType mul, dnnType add); 
    virtual ~MulAdd();
    virtual layerType_t getLayerType() { return LAYER_MULADD; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

protected:
    dnnType mul, add;
    dnnType *add_vector;
};



/**
    Avaible pooling functions (padding on tkDNN is not supported)
*/
typedef enum {
    POOLING_MAX     = 0,
    POOLING_AVERAGE = 1,                  // count for average includes padded values
    POOLING_AVERAGE_EXCLUDE_PADDING = 2   // count for average does not include padded values
} tkdnnPoolingMode_t;

/**
    Pooling layer
    currenty supported only 2d pooing (also on 3d input)
*/
class Pooling : public Layer {

public:
    int winH, winW;
    int strideH, strideW;
    int paddingH, paddingW;

    Pooling(Network *net, int winH, int winW, 
            int strideH, int strideW, tkdnnPoolingMode_t pool_mode); 
    virtual ~Pooling();
    virtual layerType_t getLayerType() { return LAYER_POOLING; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

protected:

    cudnnPoolingDescriptor_t poolingDesc;
    tkdnnPoolingMode_t pool_mode;
    dnnType *tmpInputData, *tmpOutputData;
    bool poolOn3d;
};

/**
    Softmax layer
*/
class Softmax : public Layer {

public:
    Softmax(Network *net); 
    virtual ~Softmax();
    virtual layerType_t getLayerType() { return LAYER_SOFTMAX; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
};

/**
    Route layer
    Merge a list of layers
*/
class Route : public Layer {

public:
    Route(Network *net, Layer **layers, int layers_n); 
    virtual ~Route();
    virtual layerType_t getLayerType() { return LAYER_ROUTE; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

public:
    Layer **layers;  //ids of layers to be merged
    int layers_n; //number of layers
};


/**
    Reorg layer
    Mantain same dimension but change C*H*W distribution
*/
class Reorg : public Layer {

public:
    Reorg(Network *net, int stride);
    virtual ~Reorg();
    virtual layerType_t getLayerType() { return LAYER_REORG; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

    int stride;
};


struct box {
    float x, y, w, h;
};
struct sortable_bbox {
    int index;
    int cl;
    float **probs;
};

/**
    Region layer
    Mantain same dimension but change C*H*W distribution
*/
class Region : public Layer {

public:
    Region(Network *net, int classes, int coords, int num);
    virtual ~Region();
    virtual layerType_t getLayerType() { return LAYER_REGION; };

    int classes, coords, num;
    
    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
};

class RegionInterpret {

public:
    RegionInterpret(dataDim_t input_dim, dataDim_t output_dim, 
                    int classes, int coords, int num, float thresh, const char* fname_weights);
    ~RegionInterpret();

    dataDim_t input_dim, output_dim;
    dnnType *bias_h, *bias_d; //anchors
    int classes, coords, num;
    float thresh;

    
    box *boxes;
    float **probs;
    sortable_bbox *s;
    box res_boxes[256];
    int res_boxes_n;

    box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride);
    void get_region_boxes(  float *input, int w, int h, int netw, int neth, float thresh, 
                            float **probs, box *boxes, int only_objectness, 
                            int *map, float tree_thresh, int relative); 
    void correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative);
    void interpretData(dnnType *data_h);
    void showImageResult(dnnType *input_h);
};

}
#endif //LAYER_H
