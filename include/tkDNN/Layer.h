#ifndef LAYER_H
#define LAYER_H

#include<iostream>
#include<vector>
#include "utils.h"
#include "Network.h"

namespace tk { namespace dnn {

enum layerType_t {
    LAYER_INPUT,
    LAYER_DENSE,
    LAYER_CONV2D,
    LAYER_DECONV2D,
    LAYER_DEFORMCONV2D,
    LAYER_LSTM,
    LAYER_ACTIVATION,
    LAYER_ACTIVATION_CRELU,
    LAYER_ACTIVATION_LEAKY,
    LAYER_ACTIVATION_MISH,
    LAYER_ACTIVATION_LOGISTIC,
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
    Layer(Network *net);
    virtual ~Layer();
    virtual layerType_t getLayerType() = 0;

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData) {
        std::cout<<"No infer action for this layer\n";
        return NULL;
    }
    void setFinal() { this->final = true; }
    dataDim_t input_dim, output_dim;
    dnnType *dstData = nullptr;  //where results will be putted

    int id = 0;
    bool final;        //if the layer is the final one

    std::string getLayerName() {
        layerType_t type = getLayerType();
        switch(type) {
            case LAYER_INPUT:               return "Input";
            case LAYER_DENSE:               return "Dense";
            case LAYER_CONV2D:              return "Conv2d";
            case LAYER_DECONV2D:            return "DeConv2d";
            case LAYER_DEFORMCONV2D:        return "DeformConv2d";
            case LAYER_LSTM:                return "LSTM";
            case LAYER_ACTIVATION:          return "Activation";
            case LAYER_ACTIVATION_CRELU:    return "ActivationCReLU";
            case LAYER_ACTIVATION_LEAKY:    return "ActivationLeaky";
            case LAYER_ACTIVATION_MISH:     return "ActivationMish";
            case LAYER_ACTIVATION_LOGISTIC: return "ActivationLogistic";
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
             std::string fname_weights, bool batchnorm = false, bool additional_bias = false, bool deConv = false, int groups = 1); 
    virtual ~LayerWgs();

    int inputs, outputs;
    std::string weights_path;

    dnnType *data_h, *data_d;
    dnnType *bias_h, *bias_d;

    // additional bias for DCN
    bool additional_bias;
    dnnType *bias2_h = nullptr, *bias2_d = nullptr;

    //batchnorm
    bool batchnorm;
    dnnType *power_h    = nullptr;
    dnnType *scales_h   = nullptr,   *scales_d = nullptr;
    dnnType *mean_h     = nullptr,     *mean_d = nullptr;
    dnnType *variance_h = nullptr, *variance_d = nullptr;

    //fp16
    __half *data16_h  = nullptr, *bias16_h  = nullptr;
    __half *data16_d  = nullptr, *bias16_d  = nullptr;
    __half *bias216_h = nullptr, *bias216_d = nullptr;

    __half *power16_h    = nullptr,    *power16_d = nullptr;
    __half *scales16_h   = nullptr,   *scales16_d = nullptr;
    __half *mean16_h     = nullptr,     *mean16_d = nullptr;
    __half *variance16_h = nullptr, *variance16_d = nullptr;

    void releaseHost(bool release32 = true, bool release16 = true) {
        if(release32) {
            if(    data_h != nullptr) { delete []     data_h;     data_h = nullptr; }
            if(    bias_h != nullptr) { delete []     bias_h;     bias_h = nullptr; }
            if(   bias2_h != nullptr) { delete []    bias2_h;    bias2_h = nullptr; }
            if(  scales_h != nullptr) { delete []   scales_h;   scales_h = nullptr; }
            if(    mean_h != nullptr) { delete []     mean_h;     mean_h = nullptr; }
            if(variance_h != nullptr) { delete [] variance_h; variance_h = nullptr; }
            if(   power_h != nullptr) { delete []    power_h;    power_h = nullptr; }
        }
        if(net->fp16 && release16) {
            if(    data16_h != nullptr) { delete []     data16_h;     data16_h = nullptr; }
            if(    bias16_h != nullptr) { delete []     bias16_h;     bias16_h = nullptr; }
            if(   bias216_h != nullptr) { delete []    bias216_h;    bias216_h = nullptr; }
            if(  scales16_h != nullptr) { delete []   scales16_h;   scales16_h = nullptr; }
            if(    mean16_h != nullptr) { delete []     mean16_h;     mean16_h = nullptr; }
            if(variance16_h != nullptr) { delete [] variance16_h; variance16_h = nullptr; } 
            if(   power16_h != nullptr) { delete []    power16_h;    power16_h = nullptr; }

        }
    }
    void releaseDevice(bool release32 = true, bool release16 = true) {
        if(release32) {
            if(    data_d != nullptr) { cudaFree(    data_d);     data_d = nullptr; }
            if(    bias_d != nullptr) { cudaFree(    bias_d);     bias_d = nullptr; }
            if(   bias2_d != nullptr) { cudaFree(   bias2_d);    bias2_d = nullptr; }
            if(  scales_d != nullptr) { cudaFree(  scales_d);   scales_d = nullptr; }
            if(    mean_d != nullptr) { cudaFree(    mean_d);     mean_d = nullptr; }
            if(variance_d != nullptr) { cudaFree(variance_d); variance_d = nullptr; }
        }
        if(net->fp16 && release16) {
            if(    data16_d != nullptr) { cudaFree(    data16_d);     data16_d = nullptr; }
            if(    bias16_d != nullptr) { cudaFree(    bias16_d);     bias16_d = nullptr; }
            if(   bias216_d != nullptr) { cudaFree(   bias216_d);    bias216_d = nullptr; }
            if(  scales16_d != nullptr) { cudaFree(  scales16_d);   scales16_d = nullptr; }
            if(    mean16_d != nullptr) { cudaFree(    mean16_d);     mean16_d = nullptr; }
            if(variance16_d != nullptr) { cudaFree(variance16_d); variance16_d = nullptr; } 
            if(   power16_d != nullptr) { cudaFree(   power16_d);    power16_d = nullptr; }
        }
    }
};


/**
    Input layer (it doesn't need weights)
*/
class Input : public Layer {

public:

    Input(Network *net, dataDim_t &dim, dnnType* srcData) : Layer(net) {
        input_dim = dim;
        output_dim = dim;
        dstData = srcData;
    }
    virtual ~Input() {}
    virtual layerType_t getLayerType() { return LAYER_INPUT; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData) {
        dim = output_dim;
        return dstData;
    }
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
    Available activation functions
*/
typedef enum {
    ACTIVATION_ELU     = 100,
    ACTIVATION_LEAKY   = 101,
    ACTIVATION_MISH   = 102,
    ACTIVATION_LOGISTIC   = 103
} tkdnnActivationMode_t;

/**
    Activation layer (it doesn't need weights)
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
        else if (act_mode == ACTIVATION_MISH)
            return LAYER_ACTIVATION_MISH;
        else if (act_mode == ACTIVATION_LOGISTIC)
            return LAYER_ACTIVATION_LOGISTIC;
        else
            return LAYER_ACTIVATION;
         };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

protected:
    cudnnActivationDescriptor_t activDesc;
};


/**
    Convolutional 2D layer

    WEIGHTS shape: OUTCH, INCH, KH, KW ...
    BIAS shape: OUTCH

    with BATCHNORM:
        scales:   OUTCH
        means:    OUTCH
        variance: OUTCH
*/
class Conv2d : public LayerWgs {

public:
    Conv2d( Network *net, int out_ch, int kernelH, int kernelW, 
                int strideH, int strideW, int paddingH, int paddingW,
                std::string fname_weights, bool batchnorm = false, bool deConv = false, int groups = 1, bool additional_bias=false);
    virtual ~Conv2d();
    virtual layerType_t getLayerType() { return LAYER_CONV2D; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

    int kernelH, kernelW, strideH, strideW, paddingH, paddingW;
    bool deConv, additional_bias;
    int groups;

protected:
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgoPerf_t     algo;
    cudnnConvolutionBwdDataAlgoPerf_t bwAlgo;
    cudnnTensorDescriptor_t biasTensorDesc;

    void initCUDNN(bool back = false);
    void inferCUDNN(dnnType* srcData, bool back = false);
    void*  workSpace;
    size_t ws_sizeInBytes;
};

/**
    Bidirectional LSTM layer
    ONLY BIDIRECTIONAL (TODO: more configurable)
    currently implemented as 2 inferences: forward and backward (TODO: only 1 cudnn inference)

    implementation info:
    https://github.com/jiangnanhugo/seq2seq_cuda/blob/e4dbdcfa0517c972bfd4beea9f11a5233954093c/src/rnn.cpp
    https://github.com/Jeffery-Song/mxnet-test/blob/aab666faad44011f7a67b527b5f6c960367d0422/src/operator/cudnn_rnn-inl.h
    https://stackoverflow.com/a/38737941
    https://colah.github.io/posts/2015-08-Understanding-LSTMs/

    PARAMS (numlayers*2):
        layer0:
            ( INCH, ? )    ???
            ( HIDDEN, ? )  ???
            ( HIDDEN * 8 ) ???
        layer2:
            ( INCH, ? )    ???
            ( HIDDEN, ? )  ???
            ( HIDDEN * 8 ) ???

    OUTPUT shape:
        (N, C, 1, W) ---> LSTM(HIDDEN, returnSeq=True)  ---> (N, 2*HIDDEN, 1, W)   # W is seqLength 
        (N, C, 1, W) ---> LSTM(HIDDEN, returnSeq=False) ---> (N, 2*HIDDEN, 1, 1)
*/
class LSTM : public Layer {

public:
    LSTM(Network *net, int hiddensize, bool returnSeq, std::string fname_weights);
    virtual ~LSTM();
    virtual layerType_t getLayerType() { return LAYER_LSTM; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

    const bool bidirectional = true; /**> is the net bidir */
    bool returnSeq = false;       /**> if false return only the result of last timestamp */
    int stateSize = 0; /**> number of hidden states */
    int seqLen = 0;    /**> number of timestamp */
    int numLayers = 1; /**> number of internal layers */

protected:
    cudnnRNNDescriptor_t rnnDesc;
    cudnnDropoutDescriptor_t dropoutDesc;
    dnnType  *dropout_states_, *work_space_;

    size_t workspace_byte_, dropout_byte_;
    int workspace_size_, dropout_size_;
    
    std::vector<cudnnTensorDescriptor_t> x_desc_vec_, y_desc_vec_;
    cudnnTensorDescriptor_t hx_desc_, cx_desc_;
    cudnnTensorDescriptor_t hy_desc_, cy_desc_;
    dnnType *hx_ptr, *cx_ptr, *hy_ptr, *cy_ptr;
    int stateDataDim;

    cudnnFilterDescriptor_t w_desc_;
    dnnType *w_ptr;
    dnnType *w_h;
    dnnType *wf_ptr, *wb_ptr; // params pointer forward and backward layer

    // used during inference
    dataDim_t one_output_dim; // output dim of as single inference
    dnnType *srcF, *srcB; // input of single inference 
    dnnType *dstF, *dstB_NR, *dstB; // output of single inference, dstB_NR = dstB not reversed
};


/**
    Convolutional 2D layer
*/
class DeConv2d : public Conv2d {

public:
    DeConv2d( Network *net, int out_ch, int kernelH, int kernelW,
            int strideH, int strideW, int paddingH, int paddingW,
            std::string fname_weights, bool batchnorm = false, int groups = 1) :
            Conv2d(net, out_ch, kernelH, kernelW, strideH, strideW, paddingH, paddingW, fname_weights, batchnorm, true, groups) {}
    virtual ~DeConv2d() {}
    virtual layerType_t getLayerType() { return LAYER_DECONV2D; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
};


/**
    Deformable Convolutional 2d layer
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
    Reshape(Network *net, dataDim_t new_dim); 
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
    Available pooling functions (padding on tkDNN is not supported)
*/
typedef enum {
    POOLING_MAX     = 0,
    POOLING_AVERAGE = 1,                    // count for average includes padded values
    POOLING_AVERAGE_EXCLUDE_PADDING = 2,    // count for average does not include padded values
    POOLING_MAX_FIXEDSIZE = 100             // max pool darknet fashion
} tkdnnPoolingMode_t;

/**
    Pooling layer
    currently supported only 2d pooing (also on 3d input)
*/
class Pooling : public Layer {

public:
    int winH, winW;
    int strideH, strideW;
    int paddingH, paddingW;
    bool size;
    tkdnnPoolingMode_t pool_mode;

    Pooling(Network *net, int winH, int winW, 
            int strideH, int strideW, 
            int paddingH, int paddingW,
            tkdnnPoolingMode_t pool_mode);
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
    Softmax(Network *net, const tk::dnn::dataDim_t* dim=nullptr, const cudnnSoftmaxMode_t mode=CUDNN_SOFTMAX_MODE_CHANNEL); 
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
    Route(Network *net, Layer **layers, int layers_n, int groups = 1, int group_id = 0); 
    virtual ~Route();
    virtual layerType_t getLayerType() { return LAYER_ROUTE; };

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);

public:
    static const int MAX_LAYERS = 32;
    Layer *layers[MAX_LAYERS];  //ids of layers to be merged
    int layers_n; //number of layers
    int groups;
    int group_id;
};


/**
    Reorg layer
    Maintains same dimension but change C*H*W distribution
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
    Maintains same dimension but change C*H*W distribution
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
    std::vector<float> probs;

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

    enum nmsKind_t {GREEDY_NMS=0, DIOU_NMS=1};

    Yolo(Network *net, int classes, int num, std::string fname_weights,int n_masks=3, float scale_xy=1, double nms_thresh=0.45, nmsKind_t nsm_kind=GREEDY_NMS, int new_coords=0);
    virtual ~Yolo();
    virtual layerType_t getLayerType() { return LAYER_YOLO; };

    int classes, num, n_masks, new_coords;
    dnnType *mask_h, *mask_d; //anchors
    dnnType *bias_h, *bias_d; //anchors
    float scaleXY;
    double nms_thresh;
    nmsKind_t nsm_kind; 
    std::vector<std::string> classesNames;

    virtual dnnType* infer(dataDim_t &dim, dnnType* srcData);
    int computeDetections(Yolo::detection *dets, int &ndets, int netw, int neth, float thresh, int new_coords=0);

    dnnType *predictions;

    static const int MAX_DETECTIONS = 8192*2;
    static Yolo::detection *allocateDetections(int nboxes, int classes);
    static void             mergeDetections(Yolo::detection *dets, int ndets, int classes, double nms_thresh=0.45, nmsKind_t nsm_kind=GREEDY_NMS);
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
