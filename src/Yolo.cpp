#include <iostream>

#ifdef OPENCV
    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/imgproc/imgproc.hpp>
#endif

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Yolo::Yolo(Network *net, int classes, int num) : 
    Layer(net) {

    this->classes = classes;
    this->num = num;

    // same
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c;
    output_dim.h = input_dim.h;
    output_dim.w = input_dim.w;
    output_dim.l = input_dim.l;

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

Yolo::~Yolo() {
    checkCuda( cudaFree(dstData) );
}

int entry_index(int batch, int location, int entry, 
            int classes, dataDim_t &input_dim, dataDim_t &output_dim) {
    int n =   location / (input_dim.w*input_dim.h);
    int loc = location % (input_dim.w*input_dim.h);
    return batch*output_dim.tot() + n*input_dim.w*input_dim.h*(4+classes+1) +
           entry*input_dim.w*input_dim.h + loc;
}


dnnType* Yolo::infer(dataDim_t &dim, dnnType* srcData) {

    checkCuda( cudaMemcpy(dstData, srcData, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));

    for (int b = 0; b < dim.n; ++b){
        for(int n = 0; n < num; ++n){
            int index = entry_index(b, n*dim.w*dim.h, 0, classes, input_dim, output_dim);
            activationLOGISTICForward(srcData + index, dstData + index, 2*dim.w*dim.h);
            
            index = entry_index(b, n*dim.w*dim.h, 4, classes, input_dim, output_dim);
            activationLOGISTICForward(srcData + index, dstData + index, (1+classes)*dim.w*dim.h);
        }
    }

    dim = output_dim;
    return dstData;
}

}}
