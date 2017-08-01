#ifndef NETWORK_H
#define NETWORK_H

#include "utils.h"

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

class Layer;
const int MAX_LAYERS = 256;

class Network {

public:
    Network(dataDim_t input_dim);
    virtual ~Network();

    /**
        Do inferece for every added layer
    */
    value_type* infer(dataDim_t &dim, value_type* data);

    bool addLayer(Layer *l);

    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    Layer* layers[MAX_LAYERS];  //contains layers of the net
    int num_layers; //current number of layers

    dataDim_t input_dim;
    dataDim_t getOutputDim();
};

}
#endif //NETWORK_H