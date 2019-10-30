#include<iostream>
#include "tkdnn.h"

const char *input_bin   = "../tests/simple/input.bin";
const char *c0_bin      = "../tests/simple/layers/c0.bin";
const char *c1_bin      = "../tests/simple/layers/c1.bin";
const char *d2_bin      = "../tests/simple/layers/d2.bin";
const char *output_bin  = "../tests/simple/output.bin";

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 1, 10, 10, 1);
    tk::dnn::Network net(dim);
    tk::dnn::Conv2d     l0(&net, 2, 4, 4, 2, 2, 0, 0, c0_bin);
    tk::dnn::Activation l1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     l2(&net, 4, 2, 2, 1, 1, 0, 0, c1_bin);
    tk::dnn::Activation l3(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Dense      l5(&net, 4, d2_bin);
    tk::dnn::Activation l6(&net, CUDNN_ACTIVATION_RELU);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    // Print input
    std::cout<<"\n======= INPUT =======\n";
    printDeviceVector(dim.tot(), data);
    std::cout<<"\n";

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, "simple.rt");

    dnnType *out_data, *out_data2; // cudnn output, tensorRT output

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TIMER_START
        out_data = net.infer(dim1, data);
        TIMER_STOP
        dim1.print();
    }

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        out_data2 = netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }

    std::cout<<"\n======= CUDNN =======\n";
    printDeviceVector(dim.tot(), out_data);
    std::cout<<"\n======= TENSORRT =======\n";
    printDeviceVector(dim.tot(), out_data2);

    printCenteredTitle(" CHECK RESULTS ", '=', 30);
    dnnType *out, *out_h;
    int out_dim = net.getOutputDim().tot();
    //readBinaryFile(output_bin, out_dim, &out_h, &out);
    //std::cout<<"CUDNN vs correct"; checkResult(out_dim, out_data, out);
    //std::cout<<"TRT   vs correct"; checkResult(out_dim, out_data2, out);
    std::cout<<"CUDNN vs TRT    "; checkResult(out_dim, out_data, out_data2);
    return 0;
}
