#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

MulAdd::MulAdd(Network *net, value_type mul, value_type add) : Layer(net) {

    this->mul = mul;
    this->add = add;

    int size = input_dim.tot();

    // create a vector with all value setted to add 
    value_type *add_vector_h = new value_type[size];
    for(int i=0; i<size; i++)
        add_vector_h[i] = add;

    checkCuda( cudaMalloc(&add_vector, size*sizeof(value_type)));
    checkCuda( cudaMemcpy(add_vector, add_vector_h, size*sizeof(value_type), cudaMemcpyHostToDevice));
    delete [] add_vector_h;


    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(value_type)) );
}

MulAdd::~MulAdd() {

    checkCuda( cudaFree(add_vector) );
    checkCuda( cudaFree(dstData) );
}

value_type* MulAdd::infer(dataDim_t &dim, value_type* srcData) {

    matrixMulAdd(net->cublasHandle, srcData, dstData, add_vector, input_dim.tot(), mul);
    
    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}