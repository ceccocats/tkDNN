#include<iostream>
#include "Layer.h"

int main() {

    tkDNN::Network net;
    tkDNN::dataDim_t dim(1, 8, 1, 1);
    tkDNN::Dense      d(&net, dim, 2, "../tests/dense.bin", "../tests/dense.bias.bin");
    tkDNN::Activation a(&net, d.output_dim, tkDNN::ACTIVATION_ELU);

    value_type *data;
    value_type *input_h;
    readBinaryFile("../tests/input.bin", 8, &input_h, &data);

    dim.print();
    data = d.infer(dim, data);
    dim.print();
    data = a.infer(dim, data);
    dim.print();

    printDeviceVector(dim.tot(), data);

    return 0;
}