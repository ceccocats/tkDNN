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
    tkDNN::Activation l5(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Dense      l6(&net, 10, d3_bin);
    tkDNN::Softmax    l7(&net);

    tkDNN::NetworkRT netRT(&net, "mnist.rt");

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dnnType *out_data, *out_data2;

    std::cout<<"CUDNN inference:\n"; {
        dim.print(); //print initial dimension  
        TIMER_START
        out_data = net.infer(dim, data);    
        TIMER_STOP
        dim.print();   
    }

    // Print result
    //std::cout<<"\n======= CUDNN RESULT =======\n";
    //printDeviceVector(10, out_data);
 
    tkDNN::dataDim_t dim2(1, 1, 28, 28, 1);

    std::cout<<"TENSORRT inference:\n"; {
        dim2.print();
        TIMER_START
        out_data2 = netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }

    // Print result
    //std::cout<<"\n======= TENRT RESULT =======\n";
    //printDeviceVector(10, out_data);

    std::cout<<"\n======= CHECK RESULT =======\n";
    checkResult(dim.tot(), out_data, out_data2);

 /*
    // Print real test
    std::cout<<"\n==== CHECK RESULT ====\n";
    dnnType *out;
    dnnType *out_h;
    readBinaryFile(output_bin, dim.tot(), &out_h, &out);
    printDeviceVector(dim.tot(), out);
*/ 
    return 0;
}
