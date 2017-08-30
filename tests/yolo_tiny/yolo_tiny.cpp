#include<iostream>
#include "tkdnn.h"

const char *input_bin  = "../tests/yolo_tiny/layers/input.bin";
const char *c0_bin     = "../tests/yolo_tiny/layers/c0.bin";
const char *c2_bin     = "../tests/yolo_tiny/layers/c2.bin";
const char *c4_bin     = "../tests/yolo_tiny/layers/c4.bin";
const char *c5_bin     = "../tests/yolo_tiny/layers/c5.bin";
const char *c6_bin     = "../tests/yolo_tiny/layers/c6.bin";
const char *c8_bin     = "../tests/yolo_tiny/layers/c8.bin";
const char *c10_bin    = "../tests/yolo_tiny/layers/c10.bin";
const char *c11_bin    = "../tests/yolo_tiny/layers/c11.bin";
const char *c12_bin    = "../tests/yolo_tiny/layers/c12.bin";
const char *c13_bin    = "../tests/yolo_tiny/layers/c13.bin";
const char *g14_bin    = "../tests/yolo_tiny/layers/g14.bin";
const char *output_bin = "../tests/yolo_tiny/layers/output.bin";

int main() {

    // Network layout
    tkDNN::dataDim_t dim(1, 3, 416, 416, 1);
    tkDNN::Network net(dim);

    tkDNN::Conv2d     c0 (&net, 16, 3, 3, 1, 1, 1, 1,   c0_bin, true);
    tkDNN::Activation a0 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p1 (&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c2 (&net, 32, 3, 3, 1, 1, 1, 1,   c2_bin, true);
    tkDNN::Activation a2 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p3 (&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c4 (&net, 64, 3, 3, 1, 1, 1, 1,  c4_bin, true);
    tkDNN::Activation a4 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p5 (&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c6 (&net, 128, 3, 3, 1, 1, 1, 1,  c6_bin, true);
    tkDNN::Activation a6 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p7(&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c8(&net, 256, 3, 3, 1, 1, 1, 1,  c8_bin, true);
    tkDNN::Activation a8(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p9(&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c10(&net, 512, 3, 3, 1, 1, 1, 1, c10_bin, true);
    tkDNN::Activation a10(&net, tkDNN::ACTIVATION_LEAKY);

    tkDNN::Conv2d     c11(&net, 1024, 3, 3, 1, 1, 1, 1, c11_bin, true);
    tkDNN::Activation a11(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c12(&net, 512, 3, 3, 1, 1, 1, 1, c12_bin, true);
    tkDNN::Activation a12(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c13(&net, 425, 1, 1, 1, 1, 0, 0, c13_bin, false);
    tkDNN::Region     g14(&net, 80, 4, 5);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    //print network model
    net.print();

    //convert network to tensorRT
    tkDNN::NetworkRT netRT(&net, "yolo_tiny.rt");

    dnnType *out_data, *out_data2; // cudnn output, tensorRT output

    tkDNN::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TIMER_START
        out_data = net.infer(dim1, data);    
        TIMER_STOP
        dim1.print();   
    }
 
    tkDNN::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        out_data2 = netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }

    printCenteredTitle(" CHECK RESULTS ", '=', 30);
    dnnType *out, *out_h;
    int out_dim = net.getOutputDim().tot();
    readBinaryFile(output_bin, out_dim, &out_h, &out);
    std::cout<<"CUDNN vs correct"; checkResult(out_dim, out_data, out);
    std::cout<<"TRT   vs correct"; checkResult(out_dim, out_data2, out);
    std::cout<<"CUDNN vs TRT    "; checkResult(out_dim, out_data, out_data2);
    return 0;
}
