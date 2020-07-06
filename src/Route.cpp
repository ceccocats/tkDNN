#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Route::Route(Network *net, Layer **layers, int layers_n, int groups, int group_id) : Layer(net) {

    // copy input layers
    if(layers_n > MAX_LAYERS) {
        FatalError("ROUTE: reached max number of input layers");
    }
    for(int i=0; i<layers_n; i++) {
        this->layers[i] = layers[i];
    }
    this->layers_n = layers_n;
    this->groups = groups;
    this->group_id = group_id;
    
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

    output_dim.c /= this->groups;
    input_dim = output_dim;

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

Route::~Route() {

    checkCuda( cudaFree(dstData) );
}

dnnType* Route::infer(dataDim_t &dim, dnnType* srcData) {


    int offset = 0;
    for(int i=0; i<layers_n; i++) {
        dnnType *input = layers[i]->dstData;
        int in_dim = layers[i]->output_dim.tot();
        int part_in_dim = in_dim / this->groups;
        checkCuda( cudaMemcpy(dstData + offset, input + this->group_id*part_in_dim, part_in_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice));
        offset += part_in_dim;
    }

    //update data dimensions    
    dim = output_dim;

    return dstData;
}

}}