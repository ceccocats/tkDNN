#include<iostream>
#include "tkdnn.h"

const char *input_bin  = "../tests/yolo3_tinyNM512/layers/input.bin";
const char *c0_bin     = "../tests/yolo3_tinyNM512/layers/c0.bin";
const char *c2_bin     = "../tests/yolo3_tinyNM512/layers/c2.bin";
const char *c4_bin     = "../tests/yolo3_tinyNM512/layers/c4.bin";
const char *c6_bin     = "../tests/yolo3_tinyNM512/layers/c6.bin";
const char *c8_bin     = "../tests/yolo3_tinyNM512/layers/c8.bin";
const char *c10_bin    = "../tests/yolo3_tinyNM512/layers/c10.bin";
const char *c11_bin    = "../tests/yolo3_tinyNM512/layers/c11.bin";
const char *c12_bin    = "../tests/yolo3_tinyNM512/layers/c12.bin";
const char *c13_bin    = "../tests/yolo3_tinyNM512/layers/c13.bin";
const char *c14_bin    = "../tests/yolo3_tinyNM512/layers/c14.bin";
const char *c17_bin    = "../tests/yolo3_tinyNM512/layers/c17.bin";
const char *c20_bin    = "../tests/yolo3_tinyNM512/layers/c20.bin";
const char *c21_bin    = "../tests/yolo3_tinyNM512/layers/c21.bin";
const char *g15_bin    = "../tests/yolo3_tinyNM512/layers/g15.bin";
const char *g22_bin    = "../tests/yolo3_tinyNM512/layers/g22.bin";
// const char *output_bin = "../tests/yolo3_tinyNM512/layers/output.bin";

const char *output_bin = "../tests/yolo3_tinyNM512/debug/layer22_out.bin";

int main() {
    
    // downloadWeightsifDoNotExist(input_bin, "../tests/yolo3_tinyNM512", "https://cloud.hipert.unimore.it/s/wRW9nmkibSe5HoS/download");
    
    int classes = 80;

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 512, 512, 1);
    tk::dnn::Network net(dim);
        

    tk::dnn::Conv2d     c0 (&net, 16, 3, 3, 1, 1, 1, 1,   c0_bin, true);
    tk::dnn::Activation a0 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p1 (&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c2 (&net, 32, 3, 3, 1, 1, 1, 1,   c2_bin, true);
    tk::dnn::Activation a2 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p3 (&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c4 (&net, 64, 3, 3, 1, 1, 1, 1,  c4_bin, true);
    tk::dnn::Activation a4 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p5 (&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c6 (&net, 128, 3, 3, 1, 1, 1, 1,  c6_bin, true);
    tk::dnn::Activation a6 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p7(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c8(&net, 256, 3, 3, 1, 1, 1, 1,  c8_bin, true);
    tk::dnn::Activation a8(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling    p9(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c10(&net, 512, 3, 3, 1, 1, 1, 1, c10_bin, true);
    tk::dnn::Activation a10(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Conv2d     c12(&net, 1024, 3, 3, 1, 1, 1, 1, c11_bin, true);
    tk::dnn::Activation a12(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Conv2d     c13(&net, 256, 1, 1, 1, 1, 0, 0, c12_bin, true);
    tk::dnn::Activation a13(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c14(&net, 512, 3, 3, 1, 1, 1, 1, c13_bin, true);
    tk::dnn::Activation a14(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c15(&net, 255, 1, 1, 1, 1, 0, 0, c14_bin, false);

    tk::dnn::Yolo     yolo0  (&net, classes, 2, g15_bin);

    tk::dnn::Layer *m17_layers[1] = { &a13 };
    tk::dnn::Route      m17  (&net, m17_layers, 1);
    tk::dnn::Conv2d     c18(&net, 128, 1, 1, 1, 1, 0, 0, c17_bin, true);
    tk::dnn::Activation a18(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Upsample   u19  (&net, 2);

    tk::dnn::Layer *m20_layers[2] = { &u19, &a8 };
    tk::dnn::Route      m20  (&net, m20_layers, 2);

    tk::dnn::Conv2d     c21(&net, 256, 3, 3, 1, 1, 1, 1, c20_bin, true);
    tk::dnn::Activation a21(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c22(&net, 255, 1, 1, 1, 1, 0, 0, c21_bin, false);

    tk::dnn::Yolo     yolo1  (&net, classes, 2, g22_bin);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    //print network model
    net.print();

    // convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("yolo3_tinyNM512"));

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

    printCenteredTitle(" CHECK RESULTS ", '=', 30);
    dnnType *out, *out_h;
    int out_dim = net.getOutputDim().tot();
    readBinaryFile(output_bin, out_dim, &out_h, &out);
    std::cout<<"CUDNN vs correct"; 
    int ret_cudnn = checkResult(out_dim, out_data, out) == 0 ? 0: ERROR_CUDNN;
    std::cout<<"TRT   vs correct"; 
    int ret_tensorrt = checkResult(out_dim, out_data2, out) == 0 ? 0 : ERROR_TENSORRT;
    std::cout<<"CUDNN vs TRT    "; 
    int ret_cudnn_tensorrt = checkResult(out_dim, out_data, out_data2) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
