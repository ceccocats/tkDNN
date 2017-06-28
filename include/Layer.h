#ifndef LAYER_H
#define LAYER_H

#include<iostream>
#include "utils.h"
#include "Network.h"

namespace tkDNN {

/**
    Data rapresentation beetween layers
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

    value_type* infer(dataDim_t &dim, value_type* srcData) {
        std::cout<<"No infer action for this layer\n";
        return NULL;
    }

    dataDim_t input_dim, output_dim;

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
             const char* fname_weights, const char* fname_bias); 
    virtual ~LayerWgs();

protected:
    int inputs, outputs;
    std::string weights_path, bias_path;

    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;
};

/**
    Dense (full interconnection) layer
*/
class Dense : public LayerWgs {

public:
    Dense(Network *net, dataDim_t in_dim, int out_ch, 
          const char* fname_weights, const char* fname_bias); 
    virtual ~Dense();

    value_type* infer(dataDim_t &dim, value_type* srcData);

protected:
    value_type *dstData;  //where results will be putted
};

/**
    Activation layer (it doesnt need weigths)
*/
typedef enum {
    ACTIVATION_SIGMOID = 0,
    ACTIVATION_RELU    = 1,
    ACTIVATION_TANH    = 2,
    ACTIVATION_ELU     = 100
} tkdnnActivationMode_t;

class Activation : public Layer {

public:
    Activation(Network *net, dataDim_t input_dim, tkdnnActivationMode_t act_mode); 
    virtual ~Activation();

    value_type* infer(dataDim_t &dim, value_type* srcData);

protected:
    tkdnnActivationMode_t act_mode;
    value_type *dstData;  //where results will be putted
};

}
#endif //LAYER_H