#include<iostream>
#include "tkdnn.h"

const char *input_bin   = "../tests/simple/input.bin";
const char *c0_bin      = "../tests/simple/layers/c0.bin";
const char *c1_bin      = "../tests/simple/layers/c1.bin";
const char *d2_bin      = "../tests/simple/layers/d2.bin";
const char *output_bin  = "../tests/simple/output.bin";

int main() {

    // Network layout
    tkDNN::dataDim_t dim(1, 1, 10, 10, 1);
    tkDNN::Network net(dim);
    tkDNN::Conv2d     l0(&net, 2, 4, 4, 2, 2, 0, 0, c0_bin);
    tkDNN::Activation l1(&net, CUDNN_ACTIVATION_RELU);
    tkDNN::Conv2d     l2(&net, 4, 2, 2, 1, 1, 0, 0, c1_bin);
    tkDNN::Activation l3(&net, CUDNN_ACTIVATION_RELU);
    tkDNN::Flatten    l4(&net);
    tkDNN::Dense      l5(&net, 4, d2_bin);
    tkDNN::Activation l6(&net, CUDNN_ACTIVATION_RELU);

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
    std::cout<<"\n======= RESULT =======\n";
    printDeviceVector(dim.tot(), data);

    // Print real test
    std::cout<<"\n==== CHECK RESULT ====\n";
    value_type *out;
    value_type *out_h;
    readBinaryFile(output_bin, dim.tot(), &out_h, &out);
    printDeviceVector(dim.tot(), out);
    return 0;
}
