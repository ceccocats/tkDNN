#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Route::Route(Network *net, Layer **layers, int layers_n) : Layer(net) {

    this->layers = layers;
    this->layers_n = layers_n;

    //get dims
    output_dim.l = 1;
    output_dim.c = 0;
    for(int i=0; i<layers_n; i++) {

        if(i==0) {
            output_dim.w = layers[i]->output_dim.w;
            output_dim.h = layers[i]->output_dim.h;
        } else {
            if( layers[i]->output_dim.w != output_dim.w ||
                layers[i]->output_dim.h != output_dim.h   )
                FatalError("Route Output dim missmatch");
        }
        output_dim.c += layers[i]->output_dim.c;
    }

    input_dim = output_dim;

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(value_type)) );
}

Route::~Route() {

    checkCuda( cudaFree(dstData) );
}

value_type* Route::infer(dataDim_t &dim, value_type* srcData) {


    int offset = 0;
    for(int i=0; i<layers_n; i++) {
        value_type *input = layers[i]->dstData;
        int in_dim = layers[i]->input_dim.tot();
        checkCuda( cudaMemcpy(dstData + offset, input, in_dim*sizeof(value_type), cudaMemcpyDeviceToDevice));
        offset += in_dim;
    }

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}