#include<iostream>
#include "tkdnn.h"

const char *input_bin  = "../tests/yolo/layers/input.bin";
const char *c0_bin     = "../tests/yolo/layers/c0.bin";
const char *c2_bin     = "../tests/yolo/layers/c2.bin";
const char *c4_bin     = "../tests/yolo/layers/c4.bin";
const char *c5_bin     = "../tests/yolo/layers/c5.bin";
const char *c6_bin     = "../tests/yolo/layers/c6.bin";
const char *c8_bin     = "../tests/yolo/layers/c8.bin";
const char *c9_bin     = "../tests/yolo/layers/c9.bin";
const char *c10_bin    = "../tests/yolo/layers/c10.bin";
const char *c12_bin    = "../tests/yolo/layers/c12.bin";
const char *c13_bin    = "../tests/yolo/layers/c13.bin";
const char *c14_bin    = "../tests/yolo/layers/c14.bin";
const char *c15_bin    = "../tests/yolo/layers/c15.bin";
const char *c16_bin    = "../tests/yolo/layers/c16.bin";
const char *c18_bin    = "../tests/yolo/layers/c18.bin";
const char *c19_bin    = "../tests/yolo/layers/c19.bin";
const char *c20_bin    = "../tests/yolo/layers/c20.bin";
const char *c21_bin    = "../tests/yolo/layers/c21.bin";
const char *c22_bin    = "../tests/yolo/layers/c22.bin";
const char *c23_bin    = "../tests/yolo/layers/c23.bin";
const char *c24_bin    = "../tests/yolo/layers/c24.bin";
const char *c26_bin    = "../tests/yolo/layers/c26.bin";
const char *c29_bin    = "../tests/yolo/layers/c29.bin";
const char *c30_bin    = "../tests/yolo/layers/c30.bin";
const char *g31_bin    = "../tests/yolo/layers/g31.bin";
const char *output_bin = "../tests/yolo/layers/output.bin";

int main() {

    // Network layout
    tkDNN::dataDim_t dim(1, 3, 608, 608, 1);
    tkDNN::Network net(dim);

    tkDNN::Conv2d     c0 (&net, 32, 3, 3, 1, 1, 1, 1,   c0_bin, true);
    tkDNN::Activation a0 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p1 (&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c2 (&net, 64, 3, 3, 1, 1, 1, 1,   c2_bin, true);
    tkDNN::Activation a2 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p3 (&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c4 (&net, 128, 3, 3, 1, 1, 1, 1,  c4_bin, true);
    tkDNN::Activation a4 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c5 (&net, 64, 1, 1, 1, 1, 0, 0,   c5_bin, true);
    tkDNN::Activation a5 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c6 (&net, 128, 3, 3, 1, 1, 1, 1,  c6_bin, true);
    tkDNN::Activation a6 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p7 (&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c8 (&net, 256, 3, 3, 1, 1, 1, 1,  c8_bin, true);
    tkDNN::Activation a8 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c9 (&net, 128, 1, 1, 1, 1, 0, 0,  c9_bin, true);
    tkDNN::Activation a9 (&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c10(&net, 256, 3, 3, 1, 1, 1, 1,  c10_bin, true);
    tkDNN::Activation a10(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p11(&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c12(&net, 512, 3, 3, 1, 1, 1, 1,  c12_bin, true);
    tkDNN::Activation a12(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c13(&net, 256, 1, 1, 1, 1, 0, 0,  c13_bin, true);
    tkDNN::Activation a13(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c14(&net, 512, 3, 3, 1, 1, 1, 1,  c14_bin, true);
    tkDNN::Activation a14(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c15(&net, 256, 1, 1, 1, 1, 0, 0,  c15_bin, true);
    tkDNN::Activation a15(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c16(&net, 512, 3, 3, 1, 1, 1, 1,  c16_bin, true); 
    tkDNN::Activation a16(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Pooling    p17(&net, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    tkDNN::Conv2d     c18(&net, 1024, 3, 3, 1, 1, 1, 1, c18_bin, true);
    tkDNN::Activation a18(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c19(&net, 512, 1, 1, 1, 1, 0, 0,  c19_bin, true);
    tkDNN::Activation a19(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c20(&net, 1024, 3, 3, 1, 1, 1, 1, c20_bin, true);
    tkDNN::Activation a20(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c21(&net, 512, 1, 1, 1, 1, 0, 0,  c21_bin, true);
    tkDNN::Activation a21(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c22(&net, 1024, 3, 3, 1, 1, 1, 1, c22_bin, true);
    tkDNN::Activation a22(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c23(&net, 1024, 3, 3, 1, 1, 1, 1, c23_bin, true);
    tkDNN::Activation a23(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Conv2d     c24(&net, 1024, 3, 3, 1, 1, 1, 1, c24_bin, true);
    tkDNN::Activation a24(&net, tkDNN::ACTIVATION_LEAKY);

    tkDNN::Layer *m25_layers[1] = { &a16 };
    tkDNN::Route      m25(&net, m25_layers, 1);
    tkDNN::Conv2d     c26(&net, 64, 1, 1, 1, 1, 0, 0,   c26_bin, true);
    tkDNN::Activation a26(&net, tkDNN::ACTIVATION_LEAKY);
    tkDNN::Reorg      r27(&net, 2);

    tkDNN::Layer *m28_layers[2] = { &r27, &a24 };
    tkDNN::Route      m28(&net, m28_layers, 2);

    tkDNN::Conv2d     c29(&net, 1024, 3, 3, 1, 1, 1, 1, c29_bin, true);
    tkDNN::Activation a29(&net, tkDNN::ACTIVATION_LEAKY);   
    tkDNN::Conv2d     c30(&net, 425, 1, 1, 1, 1, 0, 0,  c30_bin, false);
    tkDNN::Region     g31(&net, 80, 4, 5, 0.6f, g31_bin);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    
    //print network model
    net.print();

    //convert network to tensorRT
    tkDNN::NetworkRT netRT(&net);

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
    
    std::cout<<"\n\nDetected objects: \n";
    g31.interpretData();
    g31.showImageResult(input_h);
    return 0;
}
