#include <iostream>

#include "Layer.h"

namespace tk { namespace dnn {

Dense::Dense(Network *net, int out_ch, std::string fname_weights) : 
    LayerWgs(net, net->getOutputDim().tot(), out_ch, 1, 1, 1, fname_weights) {

    output_dim.n = 1;
    output_dim.c = out_ch;
    output_dim.h = 1;
    output_dim.w = 1;
    output_dim.l = 1;

    //allocate data for infer result
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

Dense::~Dense() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Dense::infer(dataDim_t &dim, dnnType* srcData) {

    if (dim.n != 1)
        FatalError("Not Implemented"); 
    
    int dim_x = dim.tot();
    int dim_y = output_dim.tot();

    if (dim_x != input_dim.tot())
        FatalError("Input mismatch");

    dnnType alpha = dnnType(1), beta = dnnType(1);
    // place bias into dstData
    checkCuda( cudaMemcpy(dstData, bias_d, dim_y*sizeof(dnnType), cudaMemcpyDeviceToDevice) );
    
    //do matrix multiplication
    checkERROR( cublasSgemv(net->cublasHandle, CUBLAS_OP_T,
                            dim_x, dim_y,
                            &alpha,
                            data_d, dim_x,
                            srcData, 1,
                            &beta,
                            dstData, 1) );

    //update data dimensions    
    dim.h = 1;
    dim.w = 1;
    dim.l = 1;
    dim.c = dim_y;

    return dstData;
}

}}
