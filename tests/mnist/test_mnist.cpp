#include<iostream>
#include "tkdnn.h"

const char *input_bin   = "../tests/mnist/input.bin";
const char *c0_bin      = "../tests/mnist/layers/c0.bin";
const char *c1_bin      = "../tests/mnist/layers/c1.bin";
const char *d2_bin      = "../tests/mnist/layers/d2.bin";
const char *d3_bin      = "../tests/mnist/layers/d3.bin";
const char *output_bin   = "../tests/mnist/output.bin";

int main() {

    // Network layout
    tkDNN::dataDim_t dim(1, 1, 28, 28, 1);
    tkDNN::Network net(dim);
    tkDNN::Conv2d     l0(&net, 20, 5, 5, 1, 1, 0, 0, c0_bin);
    tkDNN::Pooling    l1(&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);
    tkDNN::Conv2d     l2(&net, 50, 5, 5, 1, 1, 0, 0, c1_bin);
    tkDNN::Pooling    l3(&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);
    tkDNN::Dense      l4(&net, 500, d2_bin);
    tkDNN::Activation l5(&net, CUDNN_ACTIVATION_RELU);
    tkDNN::Dense      l6(&net, 10, d3_bin);
    tkDNN::Softmax    l7(&net);
 
    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    printDeviceVector(dim.tot(), data);
    dim.print(); //print initial dimension
    
    TIMER_START

    // Inference
    data = net.infer(dim, data);
    
    TIMER_STOP
    dim.print();   

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
