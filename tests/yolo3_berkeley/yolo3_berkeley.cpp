#include<iostream>
#include "tkdnn.h"

const char *input_bin  = "../tests/yolo3_berkeley/layers/input.bin";
const char *c0_bin     = "../tests/yolo3_berkeley/layers/c0.bin";
const char *c1_bin     = "../tests/yolo3_berkeley/layers/c1.bin";
const char *c2_bin     = "../tests/yolo3_berkeley/layers/c2.bin";
const char *c3_bin     = "../tests/yolo3_berkeley/layers/c3.bin";
const char *output_bin = "../tests/yolo3_berkeley/debug/layer3_out.bin";

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 320, 544, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d     c0 (&net, 32, 3, 3, 1, 1, 1, 1,   c0_bin, true);
    tk::dnn::Activation a0 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c1 (&net, 64, 3, 3, 2, 2, 1, 1,   c1_bin, true);
    tk::dnn::Activation a1 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c2 (&net, 32, 1, 1, 1, 1, 0, 0,   c2_bin, true);
    tk::dnn::Activation a2 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c3 (&net, 64, 3, 3, 1, 1, 1, 1,   c3_bin, true);
    tk::dnn::Activation a3 (&net, tk::dnn::ACTIVATION_LEAKY);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    
    //print network model
    net.print();


    dnnType *out_data;  // cudnn output

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TIMER_START
        out_data = net.infer(dim1, data);    
        TIMER_STOP
        dim1.print();   
    }

    printCenteredTitle(" CHECK RESULTS ", '=', 30);
    dnnType *out, *out_h;
    int out_dim = net.getOutputDim().tot();
    readBinaryFile(output_bin, out_dim, &out_h, &out);
    std::cout<<"CUDNN vs correct"; checkResult(out_dim, out_data, out);
    return 0;
}
