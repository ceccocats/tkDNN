#include<iostream>
#include "Layer.h"

const char *input_bin   = "../tests/input.bin";
const char *d0_bin      = "../tests/dense0.bin";
const char *d0_bias_bin = "../tests/dense0.bias.bin";
const char *d1_bin      = "../tests/dense1.bin";
const char *d1_bias_bin = "../tests/dense1.bias.bin";
const char *d2_bin      = "../tests/dense2.bin";
const char *d2_bias_bin = "../tests/dense2.bias.bin";

int main() {

    // Network layout
    tkDNN::Network net;
    tkDNN::dataDim_t dim(1, 512, 1, 1);
    tkDNN::Dense      d0 (&net, dim,           256, d0_bin, d0_bias_bin);
    tkDNN::Activation a0 (&net, d0.output_dim, tkDNN::ACTIVATION_ELU);
    tkDNN::Dense      d1 (&net, a0.output_dim, 32,  d1_bin, d1_bias_bin);
    tkDNN::Activation a1 (&net, d1.output_dim, tkDNN::ACTIVATION_ELU);
    tkDNN::Dense      d2 (&net, a1.output_dim, 2,   d2_bin, d2_bias_bin);
    
    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dim.print(); //print initial dimension
    
    // Inference
    data = d0.infer(dim, data); dim.print();
    data = a0.infer(dim, data); dim.print();
    data = d1.infer(dim, data); dim.print();
    data = a1.infer(dim, data); dim.print();
    data = d2.infer(dim, data); dim.print();
  
    // Print result
    printDeviceVector(dim.tot(), data);
    return 0;
}