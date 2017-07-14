#include<iostream>
#include "tkdnn.h"

const char *input_bin   = "../tests/input.bin";
const char *c0_bin      = "../tests/conv0.bin";
const char *c0_bias_bin = "../tests/conv0.bias.bin";
const char *c1_bin      = "../tests/conv1.bin";
const char *c1_bias_bin = "../tests/conv1.bias.bin";

int main() {

    // Network layout
    tkDNN::Network net;
    tkDNN::dataDim_t dim(1, 1, 10, 10, 1);
    tkDNN::Layer *l;
    l = new tkDNN::Conv2d     (&net, dim, 2, 4, 4, 2, 2, c0_bin, c0_bias_bin);
    l = new tkDNN::Activation (&net, l->output_dim, CUDNN_ACTIVATION_RELU);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 4, 2, 2, 1, 1, c1_bin, c1_bias_bin);
    l = new tkDNN::Activation (&net, l->output_dim, CUDNN_ACTIVATION_RELU);

    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    printDeviceVector(dim.tot(), data);
    dim.print(); //print initial dimension
    
    TIMER_START

    // Inference
    data = net.infer(dim, data); dim.print();
    

    TIMER_STOP

    // Print result
    printDeviceVector(dim.tot(), data);
    return 0;
}
