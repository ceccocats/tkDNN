#include<iostream>
#include "tkdnn.h"

const char *input_bin   = "mnist/input.bin";
const char *c0_bin      = "mnist/layers/c0.bin";
const char *c1_bin      = "mnist/layers/c1.bin";
const char *d2_bin      = "mnist/layers/d2.bin";
const char *d3_bin      = "mnist/layers/d3.bin";
const char *output_bin   = "mnist/output.bin";

int main() {

    downloadWeightsifDoNotExist(input_bin, "mnist", "https://cloud.hipert.unimore.it/s/2TyQkMJL3LArLAS/download");

    // Network layout
    tk::dnn::dataDim_t dim(1, 1, 28, 28, 1);
    tk::dnn::Network net(dim);
    tk::dnn::Conv2d     l0(&net, 20, 5, 5, 1, 1, 0, 0, c0_bin);
    tk::dnn::Pooling    l1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
    tk::dnn::Conv2d     l2(&net, 50, 5, 5, 1, 1, 0, 0, c1_bin);
    tk::dnn::Pooling    l3(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
    tk::dnn::Dense      l4(&net, 500, d2_bin);
    tk::dnn::Activation l5(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Dense      l6(&net, 10, d3_bin);
    tk::dnn::Softmax    l7(&net);

    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("mnist"));

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dnnType *out_data, *out_data2;

    std::cout<<"CUDNN inference:\n"; {
        dim.print(); //print initial dimension  
        TKDNN_TSTART
        out_data = net.infer(dim, data);    
        TKDNN_TSTOP
        dim.print();   
    }

    // Print result
    //std::cout<<"\n======= CUDNN RESULT =======\n";
    //printDeviceVector(10, out_data);
 
    tk::dnn::dataDim_t dim2(1, 1, 28, 28, 1);

    std::cout<<"TENSORRT inference:\n"; {
        dim2.print();
        TKDNN_TSTART
        out_data2 = netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }

    // Print result
    //std::cout<<"\n======= TENRT RESULT =======\n";
    //printDeviceVector(10, out_data);

    std::cout<<"\n======= CHECK RESULT =======\n";
    int ret_tensorrt = checkResult(dim.tot(), out_data, out_data2) == 0 ? 0 : ERROR_TENSORRT;

 /*
    // Print real test
    std::cout<<"\n==== CHECK RESULT ====\n";
    dnnType *out;
    dnnType *out_h;
    readBinaryFile(output_bin, dim.tot(), &out_h, &out);
    printDeviceVector(dim.tot(), out);
*/ 
    return ret_tensorrt;
}
