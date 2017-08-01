#ifndef LAYER_H
#define LAYER_H

#include<iostream>
#include "utils.h"
#include "Network.h"

namespace tkDNN {

/**
    Data rapresentation beetween layers
    n = batch size
    c = channels
    h = heigth (lines)
    w = width  (rows)
    l = lenght (3rd dimension)
*/
struct dataDim_t {

    int n, c, h, w, l;

    dataDim_t() : n(1), c(1), h(1), w(1), l(1) {};

    dataDim_t(int _n, int _c, int _h, int _w, int _l = 1) :
        n(_n), c(_c), h(_h), w(_w), l(_l) {};

    void print() {
        std::cout<<"Data dim: "<<n<<" "<<c<<" "<<h<<" "<<w<<" "<<l<<"\n";
    }

    int tot() {
        return n*c*h*w*l;
    }
};


/** 
    Simple layer Father class
*/
class Layer {

public:
    Layer(Network *net, dataDim_t input_dim);
    virtual ~Layer();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData) {
        std::cout<<"No infer action for this layer\n";
        return NULL;
    }

    dataDim_t input_dim, output_dim;
    value_type *dstData;  //where results will be putted

protected:
    Network *net;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;

};


/**
    Father class of all layer that need to load trained weights
*/
class LayerWgs : public Layer {

public:
    LayerWgs(Network *net, dataDim_t input_dim, 
             int inputs, int outputs, int kh, int kw, int kt, 
             const char* fname_weights, bool batchnorm = false); 
    virtual ~LayerWgs();

protected:
    int inputs, outputs;
    std::string weights_path;

    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;

    //batchnorm
    bool batchnorm;
    value_type *scales_h,   *scales_d;
    value_type *mean_h,     *mean_d;
    value_type *variance_h, *variance_d;
};


/**
    Dense (full interconnection) layer
*/
class Dense : public LayerWgs {

public:
    Dense(Network *net, dataDim_t in_dim, int out_ch, 
          const char* fname_weights); 
    virtual ~Dense();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);
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
    Activation(Network *net, dataDim_t input_dim, int act_mode); 
    virtual ~Activation();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);

protected:
    int act_mode;
    cudnnActivationDescriptor_t activDesc;
};


/**
    Convolutional 2D layer
*/
class Conv2d : public LayerWgs {

public:
    Conv2d( Network *net, dataDim_t in_dim, int out_ch, 
                int kernelH, int kernelW, int strideH, int strideW,
                int paddingH, int paddingW,
                const char* fname_weights, bool batchnorm = false); 
    virtual ~Conv2d();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);

protected:
    int kernelH, kernelW, strideH, strideW;

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
    Flatten(Network *net, dataDim_t input_dim); 
    virtual ~Flatten();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);
};


/**
    MulAdd layer
    apply a multiplication and then an addition for each data
*/
class MulAdd : public Layer {

public:
    MulAdd(Network *net, dataDim_t input_dim, value_type mul, value_type add); 
    virtual ~MulAdd();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);

protected:
    value_type mul, add;
    value_type *add_vector;
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
    Pooling(Network *net, dataDim_t input_dim, int winH, int winW, 
            int strideH, int strideW, tkdnnPoolingMode_t pool_mode); 
    virtual ~Pooling();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);

protected:

    cudnnPoolingDescriptor_t poolingDesc;

    int winH, winW;
    int strideH, strideW;
    tkdnnPoolingMode_t pool_mode;
    value_type *tmpInputData, *tmpOutputData;
    bool poolOn3d;
};

/**
    Softmax layer
*/
class Softmax : public Layer {

public:
    Softmax(Network *net, dataDim_t input_dim); 
    virtual ~Softmax();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);
};

/**
    Route layer
    Merge a list of layers
*/
class Route : public Layer {

public:
    Route(Network *net, int *layers_id, int layers_n); 
    virtual ~Route();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);

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
    Reorg(Network *net, dataDim_t input_dim, int stride);
    virtual ~Reorg();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);

protected:
    int stride;
};

/**
    Region layer
    Mantain same dimension but change C*H*W distribution
*/
class Region : public Layer {

public:
    Region(Network *net, dataDim_t input_dim, 
        int classes, int coords, int num, float thresh);
    virtual ~Region();

    virtual value_type* infer(dataDim_t &dim, value_type* srcData);

protected:
    int classes, coords, num;
    float thresh;

    int entry_index(int batch, int location, int entry);
};



}
#endif //LAYER_H
