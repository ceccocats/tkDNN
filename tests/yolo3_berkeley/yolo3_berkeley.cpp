#include<iostream>
#include "tkdnn.h"

const char *input_bin  = "../tests/yolo3_berkeley/layers/input.bin";
const char *c0_bin     = "../tests/yolo3_berkeley/layers/c0.bin";
const char *c1_bin     = "../tests/yolo3_berkeley/layers/c1.bin";
const char *c2_bin     = "../tests/yolo3_berkeley/layers/c2.bin";
const char *c3_bin     = "../tests/yolo3_berkeley/layers/c3.bin";
const char *c5_bin     = "../tests/yolo3_berkeley/layers/c5.bin";
const char *c6_bin     = "../tests/yolo3_berkeley/layers/c6.bin";
const char *c7_bin     = "../tests/yolo3_berkeley/layers/c7.bin";
const char *c9_bin     = "../tests/yolo3_berkeley/layers/c9.bin";
const char *c10_bin     = "../tests/yolo3_berkeley/layers/c10.bin";
const char *c12_bin     = "../tests/yolo3_berkeley/layers/c12.bin";
const char *c13_bin     = "../tests/yolo3_berkeley/layers/c13.bin";
const char *c14_bin     = "../tests/yolo3_berkeley/layers/c14.bin";
const char *c16_bin     = "../tests/yolo3_berkeley/layers/c16.bin";
const char *c17_bin     = "../tests/yolo3_berkeley/layers/c17.bin";
const char *c19_bin     = "../tests/yolo3_berkeley/layers/c19.bin";
const char *c20_bin     = "../tests/yolo3_berkeley/layers/c20.bin";
const char *c22_bin     = "../tests/yolo3_berkeley/layers/c22.bin";
const char *c23_bin     = "../tests/yolo3_berkeley/layers/c23.bin";
const char *c25_bin     = "../tests/yolo3_berkeley/layers/c25.bin";
const char *c26_bin     = "../tests/yolo3_berkeley/layers/c26.bin";
const char *c28_bin     = "../tests/yolo3_berkeley/layers/c28.bin";
const char *c29_bin     = "../tests/yolo3_berkeley/layers/c29.bin";
const char *c31_bin     = "../tests/yolo3_berkeley/layers/c31.bin";
const char *c32_bin     = "../tests/yolo3_berkeley/layers/c32.bin";
const char *c34_bin     = "../tests/yolo3_berkeley/layers/c34.bin";
const char *c35_bin     = "../tests/yolo3_berkeley/layers/c35.bin";
const char *output_bin = "../tests/yolo3_berkeley/debug/layer36_out.bin";

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 320, 544, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d     c0   (&net,  32, 3, 3, 1, 1, 1, 1,  c0_bin, true);
    tk::dnn::Activation a0   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c1   (&net,  64, 3, 3, 2, 2, 1, 1,  c1_bin, true);
    tk::dnn::Activation a1   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c2   (&net,  32, 1, 1, 1, 1, 0, 0,  c2_bin, true);
    tk::dnn::Activation a2   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c3   (&net,  64, 3, 3, 1, 1, 1, 1,  c3_bin, true);
    tk::dnn::Activation a3   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s4   (&net, &a1);
    tk::dnn::Conv2d     c5   (&net, 128, 3, 3, 2, 2, 1, 1,  c5_bin, true);
    tk::dnn::Activation a5   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c6   (&net,  64, 1, 1, 1, 1, 0, 0,  c6_bin, true);
    tk::dnn::Activation a6   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c7   (&net, 128, 3, 3, 1, 1, 1, 1,  c7_bin, true);
    tk::dnn::Activation a7   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s8   (&net, &a5);
    tk::dnn::Conv2d     c9   (&net,  64, 1, 1, 1, 1, 0, 0,  c9_bin, true);
    tk::dnn::Activation a9   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c10  (&net, 128, 3, 3, 1, 1, 1, 1, c10_bin, true);
    tk::dnn::Activation a10  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s11  (&net, &s8);
    tk::dnn::Conv2d     c12  (&net, 256, 3, 3, 2, 2, 1, 1, c12_bin, true);
    tk::dnn::Activation a12  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c13  (&net, 128, 1, 1, 1, 1, 0, 0, c13_bin, true);
    tk::dnn::Activation a13  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c14  (&net, 256, 3, 3, 1, 1, 1, 1, c14_bin, true);
    tk::dnn::Activation a14  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s15  (&net, &a12);

    tk::dnn::Conv2d     c16  (&net, 128, 1, 1, 1, 1, 0, 0, c16_bin, true);
    tk::dnn::Activation a16  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c17  (&net, 256, 3, 3, 1, 1, 1, 1, c17_bin, true);
    tk::dnn::Activation a17  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s18  (&net, &s15);
    tk::dnn::Conv2d     c19  (&net, 128, 1, 1, 1, 1, 0, 0, c19_bin, true);
    tk::dnn::Activation a19  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c20  (&net, 256, 3, 3, 1, 1, 1, 1, c20_bin, true);
    tk::dnn::Activation a20  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s21  (&net, &s18);
    tk::dnn::Conv2d     c22  (&net, 128, 1, 1, 1, 1, 0, 0, c22_bin, true);
    tk::dnn::Activation a22  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c23  (&net, 256, 3, 3, 1, 1, 1, 1, c23_bin, true);
    tk::dnn::Activation a23  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s24  (&net, &s21);
    tk::dnn::Conv2d     c25  (&net, 128, 1, 1, 1, 1, 0, 0, c25_bin, true);
    tk::dnn::Activation a25  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c26  (&net, 256, 3, 3, 1, 1, 1, 1, c26_bin, true);
    tk::dnn::Activation a26  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s27  (&net, &s24);
    tk::dnn::Conv2d     c28  (&net, 128, 1, 1, 1, 1, 0, 0, c28_bin, true);
    tk::dnn::Activation a28  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c29  (&net, 256, 3, 3, 1, 1, 1, 1, c29_bin, true);
    tk::dnn::Activation a29  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s30  (&net, &s27);
    tk::dnn::Conv2d     c31  (&net, 128, 1, 1, 1, 1, 0, 0, c31_bin, true);
    tk::dnn::Activation a31  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c32  (&net, 256, 3, 3, 1, 1, 1, 1, c32_bin, true);
    tk::dnn::Activation a32  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s33  (&net, &s30);
    tk::dnn::Conv2d     c34  (&net, 128, 1, 1, 1, 1, 0, 0, c34_bin, true);
    tk::dnn::Activation a34  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c35  (&net, 256, 3, 3, 1, 1, 1, 1, c35_bin, true);
    tk::dnn::Activation a35  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s36  (&net, &s33);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    
    //print network model
    net.print();


    dnnType *out_data;  // cudnn output

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TIMER_START
        out_data = net.infer(dim1, data);    
        TIMER_STOP
        dim1.print();   
    }

    printCenteredTitle(" CHECK RESULTS ", '=', 30);
    dnnType *out, *out_h;
    int out_dim = net.getOutputDim().tot();
    readBinaryFile(output_bin, out_dim, &out_h, &out);
    std::cout<<"CUDNN vs correct"; checkResult(out_dim, out_data, out);
    return 0;
}
