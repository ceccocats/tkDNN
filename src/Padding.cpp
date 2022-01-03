//
// Created by perseusdg on 03/01/22.
//

#include <iostream>
#include "Layer.h"
#include "kernels.h"

namespace tk{ namespace dnn {
    Padding::Padding(Network *net, int32_t pad_h, int32_t pad_w, tkdnnPaddingMode_t padding_mode) : Layer(net) {
        this->paddingH = pad_h;
        this->paddingW = pad_w;
        this->padding_mode = padding_mode;
        output_dim.c = input_dim.c;
        output_dim.n = input_dim.n;
        output_dim.h = input_dim.h + 2 * (this->paddingH);
        output_dim.w = input_dim.w + 2 * (this->paddingW);
        checkCuda(cudaMalloc(&dstData,output_dim.tot()*sizeof(dnnType)));
    }

    Padding::~Padding() {
        checkCuda(cudaFree(dstData));
    }
    dnnType* Padding::infer(dataDim_t &dim, float *srcData) {
        fill(dstData,output_dim.tot(),0.0);
        if(padding_mode == tkdnnPaddingMode_t::PADDING_MODE_REFLECTION)
        {
            reflection_pad2d_out_forward(paddingH, paddingW, srcData, dstData, input_dim.h, input_dim.w, input_dim.c,
                                         input_dim.n);
        }
        dim = output_dim;
        return dstData;
    }

}}
