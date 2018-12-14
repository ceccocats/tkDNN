#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

MulAdd::MulAdd(Network *net, dnnType mul, dnnType add) : Layer(net) {

    this->mul = mul;
    this->add = add;

    int size = input_dim.tot();

    // create a vector with all value setted to add 
    dnnType *add_vector_h = new dnnType[size];
    for(int i=0; i<size; i++)
        add_vector_h[i] = add;

    checkCuda( cudaMalloc(&add_vector, size*sizeof(dnnType)));
    checkCuda( cudaMemcpy(add_vector, add_vector_h, size*sizeof(dnnType), cudaMemcpyHostToDevice));
    delete [] add_vector_h;


    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(dnnType)) );
}

MulAdd::~MulAdd() {

    checkCuda( cudaFree(add_vector) );
    checkCuda( cudaFree(dstData) );
}

dnnType* MulAdd::infer(dataDim_t &dim, dnnType* srcData) {

    matrixMulAdd(net->cublasHandle, srcData, dstData, add_vector, input_dim.tot(), mul);
    
    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}}