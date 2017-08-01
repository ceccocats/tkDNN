#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Region::Region(Network *net, dataDim_t input_dim, 
               int classes, int coords, int num, float thresh) : 
    Layer(net, input_dim) {

    this->classes = classes;
    this->coords = coords;
    this->num = num;
    this->thresh = thresh;
    
    // same
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c;
    output_dim.h = input_dim.h;
    output_dim.w = input_dim.w;
    output_dim.l = input_dim.l;

    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(value_type)) );
}

Region::~Region() {
    checkCuda( cudaFree(dstData) );
}

int Region::entry_index(int batch, int location, int entry) {
    int n =   location / (input_dim.w*input_dim.h);
    int loc = location % (input_dim.w*input_dim.h);
    return batch*output_dim.tot() + n*input_dim.w*input_dim.h*(coords+classes+1) + entry*input_dim.w*input_dim.h + loc;
}

value_type* Region::infer(dataDim_t &dim, value_type* srcData) {

    checkCuda( cudaMemcpy(dstData, srcData, dim.tot()*sizeof(value_type), cudaMemcpyDeviceToDevice));

    for (int b = 0; b < dim.n; ++b){
        for(int n = 0; n < num; ++n){
            int index = entry_index(b, n*dim.w*dim.h, 0);
            activationLOGISTICForward(srcData + index, dstData + index, 2*dim.w*dim.h);
            
            index = entry_index(b, n*dim.w*dim.h, coords);
            activationLOGISTICForward(srcData + index, dstData + index, dim.w*dim.h);
        }
    }

    //softmax start
    int index = entry_index(0, 0, coords + 1);
    softmaxForward(srcData + index, classes, output_dim.n*num, output_dim.tot()/num, 
                   output_dim.w*output_dim.h, 1, output_dim.w*output_dim.h, 1, dstData + index);

    dim = output_dim;
    return dstData;
}

}
