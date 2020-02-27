#include<iostream>
#include "tkdnn.h"

const char *input_bin   = "../tests/simple/input.bin";
const char *c0_bin      = "../tests/simple/layers/conv1d_1.bin";
const char *l1_bin      = "../tests/simple/layers/bidirectional_1.bin";
const char *l2_bin      = "../tests/simple/layers/bidirectional_2.bin";
const char *output_bin  = "../tests/simple/output.bin";

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 8, 1, 3);
    tk::dnn::Network net(dim);
    tk::dnn::Conv2d     l0(&net, 4, 1, 2, 1, 1, 0, 0, c0_bin);
    tk::dnn::LSTM       l1(&net, 5, true, l1_bin);
    tk::dnn::LSTM       l2(&net, 5, false, l2_bin);

    net.print();

    // Load input
    dnnType *data;
    dnnType *input_h;
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
    dnnType *out;
    dnnType *out_h;
    readBinaryFile(output_bin, dim.tot(), &out_h, &out);
    printDeviceVector(dim.tot(), out);
    return 0;
}
