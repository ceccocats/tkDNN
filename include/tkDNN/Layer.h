#ifndef LAYER_H
#define LAYER_H

#include<iostream>
#include<vector>
#include "utils.h"
#include "Network.h"

namespace tk { namespace dnn {

enum layerType_t {
    LAYER_DENSE,
    LAYER_CONV2D,
    LAYER_DECONV2D,
    LAYER_DEFORMCONV2D,
    LAYER_ACTIVATION,
    LAYER_ACTIVATION_CRELU,
    LAYER_ACTIVATION_LEAKY,
    LAYER_FLATTEN,
    LAYER_RESHAPE,
    LAYER_MULADD,
    LAYER_POOLING,
    LAYER_SOFTMAX,
    LAYER_ROUTE,
    LAYER_REORG,
    LAYER_SHORTCUT,
    LAYER_UPSAMPLE,
    LAYER_REGION,
    LAYER_YOLO
};

#define TKDNN_BN_MIN_EPSILON 1e-5

/** 
    Simple layer Father class
*/
class Layer {

public:
    Layer(Network *net, bool final = false);
    virtual ~Layer();
    virtual layerType_t getLayerType() = 0;

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData) {
        std::cout<<"No infer action for this layer\n";
        return NULL;
    }

    dataDim_t input_dim, output_dim;
    dnnType *dstData;  //where results will be putted

    int id = 0;
    bool final;        //if the layer is the final one

    std::string getLayerName() {
        layerType_t type = getLayerType();
        switch(type) {
            case LAYER_DENSE:               return "Dense";
            case LAYER_CONV2D:              return "Conv2d";
            case LAYER_DECONV2D:            return "DeConv2d";
            case LAYER_DEFORMCONV2D:        return "DeformConv2d";
            case LAYER_ACTIVATION:          return "Activation";
            case LAYER_ACTIVATION_CRELU:    return "ActivationCReLU";
            case LAYER_ACTIVATION_LEAKY:    return "ActivationLeaky";
            case LAYER_FLATTEN:             return "Flatten";
            case LAYER_RESHAPE:             return "Reshape";
            case LAYER_MULADD:              return "MulAdd";
            case LAYER_POOLING:             return "Pooling";
            case LAYER_SOFTMAX:             return "Softmax";
            case LAYER_ROUTE:               return "Route";            
            case LAYER_REORG:               return "Reorg";
            case LAYER_SHORTCUT:            return "Shortcut";
            case LAYER_UPSAMPLE:            return "Upsample";
            case LAYER_REGION:              return "Region";
            case LAYER_YOLO:                return "Yolo";
            default:                        return "unknown";
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
             std::string fname_weights, bool batchnorm = false, bool additional_bias = false, bool final = false, bool deConv = false, int groups = 1); 
    virtual ~LayerWgs();

    int inputs, outputs;
    std::string weights_path;

    dnnType *data_h, *data_d;
    dnnType *bias_h, *bias_d;

    // additional bias for DCN
    bool additional_bias;
    dnnType *bias2_h, *bias2_d;

    //batchnorm
    bool batchnorm;
    dnnType *power_h;
    dnnType *scales_h,   *scales_d;
    dnnType *mean_h,     *mean_d;
    dnnType *variance_h, *variance_d;

    //fp16
    __half *data16_h, *bias16_h;
    __half *data16_d, *bias16_d;
    __half *bias216_h, *bias216_d;

    __half *power16_h,    *power16_d;
    __half *scales16_h,   *scales16_d;
    __half *mean16_h,     *mean16_d;
    __half *variance16_h, *variance16_d;
};


/**
    Dense (full interconnection) layer
*/
class Dense : public LayerWgs {

public:
    Dense(Network *net, int out_ch, std::string fname_weights); 
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
    float ceiling;

    Activation(Network *net, int act_mode, const float ceiling=0.0); 
    virtual ~Activation();
    virtual layerType_t getLayerType() { 
        if(act_mode == CUDNN_ACTIVATION_CLIPPED_RELU)
            return LAYER_ACTIVATION_CRELU;
        else if (act_mode == ACTIVATION_LEAKY)
            return LAYER_ACTIVATION_LEAKY;
        else
            return LAYER_ACTIVATION;
         };

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
                std::string fname_weights, bool batchnorm = false, bool deConv = false, bool final = false, int groups = 1, bool additional_bias=false);
    virtual ~Conv2d();
    virtual layerType_t getLayerType() { return LAYER_CONV2D; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

    int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
    bool deConv, additional_bias;
    int groups;

protected:
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t     algo;
    cudnnConvolutionBwdDataAlgo_t bwAlgo;
    cudnnTensorDescriptor_t biasTensorDesc;

    void initCUDNN(bool back = false);
    void inferCUDNN(dnnType* srcData, bool back = false);
    void*  workSpace;
    size_t ws_sizeInBytes;
};


/**
    Convolutional 2D layer
*/
class DeConv2d : public Conv2d {

public:
    DeConv2d( Network *net, int out_ch, int kernelH, int kernelW,
            int strideH, int strideW, int paddingH, int paddingW,
            std::string fname_weights, bool batchnorm = false, int groups = 1) :
            Conv2d(net, out_ch, kernelH, kernelW, strideH, strideW, paddingH, paddingW, fname_weights, batchnorm, true, false, groups) {}
    virtual ~DeConv2d() {}
    virtual layerType_t getLayerType() { return LAYER_DECONV2D; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
};


/**
    Deformable Convolutionl 2d layer
*/  
class DeformConv2d : public LayerWgs {

public:
    DeformConv2d( Network *net, int out_ch, int deformable_group, int kernelH, int kernelW,
                int strideH, int strideW, int paddingH, int paddingW,
                std::string d_fname_weights, std::string fname_weights, bool batchnorm);
    virtual ~DeformConv2d();
    virtual layerType_t getLayerType() { return LAYER_DEFORMCONV2D; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
    tk::dnn::Conv2d *preconv; 
    int out_ch;
    int deformableGroup;
    int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
    dnnType *ones_d1;
    dnnType *ones_d2;
    int chunk_dim;
    dnnType *offset, *mask;
    dnnType *output_conv;

    cublasStatus_t stat;
    cublasHandle_t handle;

protected:

    cudnnTensorDescriptor_t biasTensorDesc;
    void initCUDNN();

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
    Reshape layer
*/
class Reshape : public Layer {

public:
    Reshape(Network *net, dataDim_t new_dim, bool final=false); 
    virtual ~Reshape();
    virtual layerType_t getLayerType() { return LAYER_RESHAPE; };

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
    bool maxpoolfixedsize;
    tkdnnPoolingMode_t pool_mode;

    Pooling(Network *net, int winH, int winW, 
            int strideH, int strideW, 
            int paddingH = 0, int paddingW = 0,
            tkdnnPoolingMode_t pool_mode = POOLING_MAX, bool final = false, bool test=false); 
    virtual ~Pooling();
    virtual layerType_t getLayerType() { return LAYER_POOLING; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

protected:

    cudnnPoolingDescriptor_t poolingDesc;
    dnnType *tmpInputData, *tmpOutputData;
    bool poolOn3d;
};

/**
    Softmax layer
*/
class Softmax : public Layer {

public:
    Softmax(Network *net, const tk::dnn::dataDim_t* dim=nullptr, bool final=false, const cudnnSoftmaxMode_t mode=CUDNN_SOFTMAX_MODE_CHANNEL); 
    virtual ~Softmax();
    virtual layerType_t getLayerType() { return LAYER_SOFTMAX; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
    dataDim_t dim;
    cudnnSoftmaxMode_t mode;
};

/**
    Route layer
    Merge a list of layers
*/
class Route : public Layer {

public:
    Route(Network *net, Layer **layers, int layers_n, bool final=false); 
    virtual ~Route();
    virtual layerType_t getLayerType() { return LAYER_ROUTE; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

public:
    static const int MAX_INPUT_LAYERS = 16;
    Layer *layers[MAX_INPUT_LAYERS];  //ids of layers to be merged
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

/**
    Shortcut layer
    sum with stride another layer
*/
class Shortcut : public Layer {

public:
    Shortcut(Network *net, Layer *backLayer); 
    virtual ~Shortcut();
    virtual layerType_t getLayerType() { return LAYER_SHORTCUT; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

public:
    Layer *backLayer;
};

/**
    Upsample layer
    Mantain same dimension but change C*H*W distribution
*/
class Upsample : public Layer {

public:
    Upsample(Network *net, int stride);
    virtual ~Upsample();
    virtual layerType_t getLayerType() { return LAYER_UPSAMPLE; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

    int stride;
    bool reverse;
};

struct box {
    int cl;
    float x, y, w, h;
    float prob;

    void print() 
    {
        std::cout<<"x: "<<x<<"\ty: "<<y<<"\tw: "<<w<<"\th: "<<h<<"\tcl: "<<cl<<"\tprob: "<<prob<<std::endl;
    }
};
struct sortable_bbox {
    int index;
    int cl;
    float **probs;
};

/**
    Yolo3 layer
*/
class Yolo : public Layer {

public:
    struct box {
        float x, y, w, h;
    };

    struct detection{
        Yolo::box bbox;
        int classes;
        float *prob;
        float *mask;
        float objectness;
        int sort_class;
    };

    Yolo(Network *net, int classes, int num, std::string fname_weights, int n_masks=3);
    virtual ~Yolo();
    virtual layerType_t getLayerType() { return LAYER_YOLO; };

    int classes, num, n_masks;
    dnnType *mask_h, *mask_d; //anchors
    dnnType *bias_h, *bias_d; //anchors
    std::vector<std::string> classesNames;

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
    int computeDetections(Yolo::detection *dets, int &ndets, int netw, int neth, float thresh);

    dnnType *predictions;

    static const int MAX_DETECTIONS = 1024;
    static Yolo::detection *allocateDetections(int nboxes, int classes);
    static void             mergeDetections(Yolo::detection *dets, int ndets, int classes);
};

/**
    Region layer
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
                    int classes, int coords, int num, float thresh, std::string fname_weights);
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
    void interpretData(dnnType *data_h, int imageW = 0, int imageH = 0);
    void showImageResult(dnnType *input_h);

    static float box_iou(box a, box b);
};

}}
#endif //LAYER_H
