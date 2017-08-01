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
const char *output_bin = "../tests/yolo/layers/output.bin";

int main() {

    // Network layout
    tkDNN::Network net;
    tkDNN::dataDim_t dim(1, 3, 608, 608, 1);
    tkDNN::Layer *l;
    l = new tkDNN::Conv2d     (&net, dim,           32, 3, 3, 1, 1, 1, 1,   c0_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Pooling    (&net, l->output_dim, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    l = new tkDNN::Conv2d     (&net, l->output_dim, 64, 3, 3, 1, 1, 1, 1,   c2_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Pooling    (&net, l->output_dim, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    l = new tkDNN::Conv2d     (&net, l->output_dim, 128, 3, 3, 1, 1, 1, 1,  c4_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 64, 1, 1, 1, 1, 0, 0,   c5_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 128, 3, 3, 1, 1, 1, 1,  c6_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Pooling    (&net, l->output_dim, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    l = new tkDNN::Conv2d     (&net, l->output_dim, 256, 3, 3, 1, 1, 1, 1,  c8_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 128, 1, 1, 1, 1, 0, 0,  c9_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 256, 3, 3, 1, 1, 1, 1,  c10_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Pooling    (&net, l->output_dim, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    l = new tkDNN::Conv2d     (&net, l->output_dim, 512, 3, 3, 1, 1, 1, 1,  c12_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 256, 1, 1, 1, 1, 0, 0,  c13_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 512, 3, 3, 1, 1, 1, 1,  c14_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 256, 1, 1, 1, 1, 0, 0,  c15_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 512, 3, 3, 1, 1, 1, 1,  c16_bin, true); 
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);                //29
    l = new tkDNN::Pooling    (&net, l->output_dim, 2, 2, 2, 2, tkDNN::POOLING_MAX);

    l = new tkDNN::Conv2d     (&net, l->output_dim, 1024, 3, 3, 1, 1, 1, 1, c18_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 512, 1, 1, 1, 1, 0, 0,  c19_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 1024, 3, 3, 1, 1, 1, 1, c20_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 512, 1, 1, 1, 1, 0, 0,  c21_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 1024, 3, 3, 1, 1, 1, 1, c22_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 1024, 3, 3, 1, 1, 1, 1, c23_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 1024, 3, 3, 1, 1, 1, 1, c24_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);                //44

    int rlayers[1] = {29};
    l = new tkDNN::Route      (&net, rlayers, 1);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 64, 1, 1, 1, 1, 0, 0,   c26_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);
    l = new tkDNN::Reorg      (&net, l->output_dim, 2);                                      //48

    int rlayers2[2] = {48,44};
    l = new tkDNN::Route      (&net, rlayers2, 2);

    l = new tkDNN::Conv2d     (&net, l->output_dim, 1024, 3, 3, 1, 1, 1, 1, c29_bin, true);
    l = new tkDNN::Activation (&net, l->output_dim, tkDNN::ACTIVATION_LEAKY);   
    l = new tkDNN::Conv2d     (&net, l->output_dim, 425, 1, 1, 1, 1, 0, 0,  c30_bin, false);

    l = new tkDNN::Region     (&net, l->output_dim, 80, 4, 5, 0.6f);

    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dim.print(); //print initial dimension
    
    TIMER_START

    // Inference
    data = net.infer(dim, data);
    
    TIMER_STOP
    dim.print();   
 
    // Print real test
    std::cout<<"\n==== CHECK RESULT ====\n";
    value_type *out;
    value_type *out_h;
    readBinaryFile(output_bin, dim.tot(), &out_h, &out);
    int diff = checkResult(dim.tot(), data, out);
    printf("Output diffs: %d\n", diff);
    return 0;
}
