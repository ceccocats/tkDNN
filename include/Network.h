#ifndef NETWORK_H
#define NETWORK_H

#include "utils.h"

namespace tkDNN {

struct dataDim_t;
class Layer;
const int MAX_LAYERS = 256;

class Network {

public:
    Network();
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

private:
    Layer* layers[MAX_LAYERS];  //contains layers of the net
    int num_layers; //current number of layers
};

}
#endif //NETWORK_H