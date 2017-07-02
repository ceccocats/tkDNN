#include<iostream>
#include "tkdnn.h"

const char *input_bin   = "../tests/input.bin";
const char *c0_bin      = "../tests/conv0.bin";
const char *c0_bias_bin = "../tests/conv0.bias.bin";
const char *c1_bin      = "../tests/conv1.bin";
const char *c1_bias_bin = "../tests/conv1.bias.bin";
const char *c2_bin      = "../tests/conv2.bin";
const char *c2_bias_bin = "../tests/conv2.bias.bin";
const char *d3_bin      = "../tests/dense3.bin";
const char *d3_bias_bin = "../tests/dense3.bias.bin";
const char *d4_bin      = "../tests/dense4.bin";
const char *d4_bias_bin = "../tests/dense4.bias.bin";
const char *d5_bin      = "../tests/dense5.bin";
const char *d5_bias_bin = "../tests/dense5.bias.bin";

int main() {

    // Network layout
    tkDNN::Network net;
    tkDNN::dataDim_t dim(1, 1, 100, 100, 4);
    tkDNN::Layer *l;
    l = new tkDNN::MulAdd     (&net, dim, 2, -1);
    l = new tkDNN::Conv3d     (&net, l->output_dim, 16, 8, 8, 2, 4, 4, 1,    c0_bin, c0_bias_bin);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_ELU);
    l = new tkDNN::Pooling    (&net, l->output_dim, 2, 2, 2, 2, tkDNN::POOLING_AVERAGE);
    l = new tkDNN::Conv3d     (&net, l->output_dim, 16, 4, 4, 2, 2, 2, 1,    c1_bin, c1_bias_bin);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_ELU);
    l = new tkDNN::Conv3d     (&net, l->output_dim, 24, 3, 3, 2, 1, 1, 1,    c2_bin, c2_bias_bin);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_ELU);
    l = new tkDNN::Flatten    (&net, l->output_dim);
    l = new tkDNN::Dense      (&net, l->output_dim, 256,                     d3_bin, d3_bias_bin);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_ELU);
    l = new tkDNN::Dense      (&net, l->output_dim, 32,                      d4_bin, d4_bias_bin);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_RELU);
    l = new tkDNN::Dense      (&net, l->output_dim, 2,                       d5_bin, d5_bias_bin);


    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dim.print(); //print initial dimension
    
    TIMER_START

    // Inference
    data = net.infer(dim, data); dim.print();

    TIMER_STOP

    // Print result
    printDeviceVector(dim.tot(), data);
    return 0;
}