#include<iostream>
#include "tkdnn.h"

const char *input_bin  = "../tests/yolo-tiny/layers/input.bin";
const char *c0_bin     = "../tests/yolo-tiny/layers/c0.bin";
const char *c2_bin     = "../tests/yolo-tiny/layers/c2.bin";
const char *c4_bin     = "../tests/yolo-tiny/layers/c4.bin";
const char *c5_bin     = "../tests/yolo-tiny/layers/c5.bin";
const char *c6_bin     = "../tests/yolo-tiny/layers/c6.bin";
const char *c8_bin     = "../tests/yolo-tiny/layers/c8.bin";
const char *c10_bin    = "../tests/yolo-tiny/layers/c10.bin";
const char *c12_bin    = "../tests/yolo-tiny/layers/c12.bin";
const char *c13_bin    = "../tests/yolo-tiny/layers/c13.bin";
const char *c14_bin    = "../tests/yolo-tiny/layers/c14.bin";
const char *g15_bin    = "../tests/yolo-tiny/layers/g15.bin";
const char *output_bin = "../tests/yolo-tiny/layers/outputLEL.bin";

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
    //tkDNN::Pooling    p11(&net, 2, 2, 1, 1, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c12(&net, 1024, 3, 3, 1, 1, 1, 1, c12_bin, true);
    tkDNN::Activation a12(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c13(&net, 1024, 3, 3, 1, 1, 1, 1, c13_bin, true);
    tkDNN::Activation a13(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c14(&net, 125, 1, 1, 1, 1, 0, 0, c14_bin, false);
    tkDNN::Region     g15(&net, 20, 4, 5, 0.6f, g15_bin);

    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    //convert network to tensorRT
    tkDNN::NetworkRT netRT(&net);

    value_type *out_data, *out_data2; // cudnn output, tensorRT output

    tkDNN::dataDim_t dim1 = dim; //input dim
    std::cout<<"\n==== CUDNN inference =======\n"; {
        dim1.print();
        TIMER_START
        out_data = net.infer(dim1, data);    
        TIMER_STOP
        dim1.print();   
    }
 
    tkDNN::dataDim_t dim2 = dim;
    std::cout<<"\n==== TENSORRT inference ====\n"; {
        dim2.print();
        TIMER_START
        out_data2 = netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }

    std::cout<<"\n======= CHECK RESULT =======\n";
    value_type *out, *out_h;
    int out_dim = net.getOutputDim().tot();
    readBinaryFile(output_bin, out_dim, &out_h, &out);
    std::cout<<"CUDNN vs correct"; checkResult(out_dim, out_data, out);
    std::cout<<"TRT   vs correct"; checkResult(out_dim, out_data2, out);
    std::cout<<"CUDNN vs TRT    "; checkResult(out_dim, out_data, out_data2);

    std::cout<<"\n\nDetected objects: \n";
    g15.interpretData();
    return 0;
}
