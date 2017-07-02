#include<iostream>
#include "Layer.h"

const char *input_bin   = "../tests/input.bin";
const char *c0_bin      = "../tests/conv0.bin";
const char *c0_bias_bin = "../tests/conv0.bias.bin";
const char *c1_bin      = "../tests/conv1.bin";
const char *c1_bias_bin = "../tests/conv1.bias.bin";
const char *d2_bin      = "../tests/dense2.bin";
const char *d2_bias_bin = "../tests/dense2.bias.bin";

int main() {

    // Network layout
    tkDNN::Network net;
    tkDNN::dataDim_t dim(1, 1, 10, 10);
    tkDNN::Conv2d     c0 (&net, dim,           2, 4, 4, 2, 2, c0_bin, c0_bias_bin);
    tkDNN::Activation a0 (&net, c0.output_dim, tkDNN::ACTIVATION_ELU);
    tkDNN::Conv2d     c1 (&net, a0.output_dim, 4, 2, 2, 1, 1, c1_bin, c1_bias_bin);
    tkDNN::Activation a1 (&net, c1.output_dim, tkDNN::ACTIVATION_RELU);

    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dim.print(); //print initial dimension
    
    TIMER_START

    // Inference
    data = net.infer(dim, data); dim.print();

    /*
    //old Inference method
    data = c0.infer(dim, data); dim.print();
    data = a0.infer(dim, data); dim.print();
    data = c1.infer(dim, data); dim.print();
    data = a1.infer(dim, data); dim.print();
    */
    TIMER_STOP
    // Print result
    printDeviceVector(dim.tot(), data);
    return 0;
}