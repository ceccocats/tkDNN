#ifndef NETWORK_H
#define NETWORK_H

#include "utils.h"

namespace tkDNN {

class Network {

public:
    Network();
    virtual ~Network();

    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
};

}
#endif //NETWORK_H