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
    tkDNN::MulAdd     m0 (&net, dim, 2, -1);
    tkDNN::Conv3d     c0 (&net, m0.output_dim, 16, 8, 8, 2, 4, 4, 1,    c0_bin, c0_bias_bin);
    tkDNN::Activation a0 (&net, c0.output_dim, tkDNN::ACTIVATION_ELU);
    tkDNN::Pooling    p0 (&net, a0.output_dim, 2, 2, 2, 2, tkDNN::POOLING_AVERAGE);
    tkDNN::Conv3d     c1 (&net, p0.output_dim, 16, 4, 4, 2, 2, 2, 1,    c1_bin, c1_bias_bin);
    tkDNN::Activation a1 (&net, c1.output_dim, tkDNN::ACTIVATION_ELU);
    tkDNN::Conv3d     c2 (&net, a1.output_dim, 24, 3, 3, 2, 1, 1, 1,    c2_bin, c2_bias_bin);
    tkDNN::Activation a2 (&net, c2.output_dim, tkDNN::ACTIVATION_ELU);
    tkDNN::Flatten    f2 (&net, a2.output_dim);
    tkDNN::Dense      d3 (&net, f2.output_dim, 256,                     d3_bin, d3_bias_bin);
    tkDNN::Activation a3 (&net, d3.output_dim, tkDNN::ACTIVATION_ELU);
    tkDNN::Dense      d4 (&net, a3.output_dim, 32,                      d4_bin, d4_bias_bin);
    tkDNN::Activation a4 (&net, d4.output_dim, tkDNN::ACTIVATION_RELU);
    tkDNN::Dense      d5 (&net, a4.output_dim, 2,                       d5_bin, d5_bias_bin);


    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dim.print(); //print initial dimension
    
    TIMER_START

    // Inference
    data = m0.infer(dim, data); dim.print();
    data = c0.infer(dim, data); dim.print();
    data = a0.infer(dim, data); dim.print();
    data = p0.infer(dim, data); dim.print();
    data = c1.infer(dim, data); dim.print();
    data = a1.infer(dim, data); dim.print();
    data = c2.infer(dim, data); dim.print();
    data = a2.infer(dim, data); dim.print();
    data = f2.infer(dim, data); dim.print();
    data = d3.infer(dim, data); dim.print();
    data = a3.infer(dim, data); dim.print();
    data = d4.infer(dim, data); dim.print();
    data = a4.infer(dim, data); dim.print();
    data = d5.infer(dim, data); dim.print();

    TIMER_STOP

    // Print result
    printDeviceVector(dim.tot(), data);
    return 0;
}