#include<iostream>
#include "tkdnn.h"

const char *input_bin  = "../tests/yolo_relu/layers/input.bin";
const char *c0_bin     = "../tests/yolo_relu/layers/c0.bin";
const char *c2_bin     = "../tests/yolo_relu/layers/c2.bin";
const char *c4_bin     = "../tests/yolo_relu/layers/c4.bin";
const char *c5_bin     = "../tests/yolo_relu/layers/c5.bin";
const char *c6_bin     = "../tests/yolo_relu/layers/c6.bin";
const char *c8_bin     = "../tests/yolo_relu/layers/c8.bin";
const char *c9_bin     = "../tests/yolo_relu/layers/c9.bin";
const char *c10_bin    = "../tests/yolo_relu/layers/c10.bin";
const char *c12_bin    = "../tests/yolo_relu/layers/c12.bin";
const char *c13_bin    = "../tests/yolo_relu/layers/c13.bin";
const char *c14_bin    = "../tests/yolo_relu/layers/c14.bin";
const char *c15_bin    = "../tests/yolo_relu/layers/c15.bin";
const char *c16_bin    = "../tests/yolo_relu/layers/c16.bin";
const char *c18_bin    = "../tests/yolo_relu/layers/c18.bin";
const char *c19_bin    = "../tests/yolo_relu/layers/c19.bin";
const char *c20_bin    = "../tests/yolo_relu/layers/c20.bin";
const char *c21_bin    = "../tests/yolo_relu/layers/c21.bin";
const char *c22_bin    = "../tests/yolo_relu/layers/c22.bin";
const char *c23_bin    = "../tests/yolo_relu/layers/c23.bin";
const char *c24_bin    = "../tests/yolo_relu/layers/c24.bin";
const char *c26_bin    = "../tests/yolo_relu/layers/c26.bin";
const char *c29_bin    = "../tests/yolo_relu/layers/c29.bin";
const char *c30_bin    = "../tests/yolo_relu/layers/c30.bin";
const char *g31_bin    = "../tests/yolo_relu/layers/g31.bin";
const char *output_bin = "../tests/yolo_relu/layers/output.bin";

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 608, 608, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d     c0 (&net, 32, 3, 3, 1, 1, 1, 1,   c0_bin, true);
    tk::dnn::Activation a0 (&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Pooling    p1 (&net, 2, 2, 2, 2, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c2 (&net, 64, 3, 3, 1, 1, 1, 1,   c2_bin, true);
    tk::dnn::Activation a2 (&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Pooling    p3 (&net, 2, 2, 2, 2, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c4 (&net, 128, 3, 3, 1, 1, 1, 1,  c4_bin, true);
    tk::dnn::Activation a4 (&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c5 (&net, 64, 1, 1, 1, 1, 0, 0,   c5_bin, true);
    tk::dnn::Activation a5 (&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c6 (&net, 128, 3, 3, 1, 1, 1, 1,  c6_bin, true);
    tk::dnn::Activation a6 (&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Pooling    p7 (&net, 2, 2, 2, 2, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c8 (&net, 256, 3, 3, 1, 1, 1, 1,  c8_bin, true);
    tk::dnn::Activation a8 (&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c9 (&net, 128, 1, 1, 1, 1, 0, 0,  c9_bin, true);
    tk::dnn::Activation a9 (&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c10(&net, 256, 3, 3, 1, 1, 1, 1,  c10_bin, true);
    tk::dnn::Activation a10(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Pooling    p11(&net, 2, 2, 2, 2, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c12(&net, 512, 3, 3, 1, 1, 1, 1,  c12_bin, true);
    tk::dnn::Activation a12(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c13(&net, 256, 1, 1, 1, 1, 0, 0,  c13_bin, true);
    tk::dnn::Activation a13(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c14(&net, 512, 3, 3, 1, 1, 1, 1,  c14_bin, true);
    tk::dnn::Activation a14(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c15(&net, 256, 1, 1, 1, 1, 0, 0,  c15_bin, true);
    tk::dnn::Activation a15(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c16(&net, 512, 3, 3, 1, 1, 1, 1,  c16_bin, true); 
    tk::dnn::Activation a16(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Pooling    p17(&net, 2, 2, 2, 2, tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d     c18(&net, 1024, 3, 3, 1, 1, 1, 1, c18_bin, true);
    tk::dnn::Activation a18(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c19(&net, 512, 1, 1, 1, 1, 0, 0,  c19_bin, true);
    tk::dnn::Activation a19(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c20(&net, 1024, 3, 3, 1, 1, 1, 1, c20_bin, true);
    tk::dnn::Activation a20(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c21(&net, 512, 1, 1, 1, 1, 0, 0,  c21_bin, true);
    tk::dnn::Activation a21(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c22(&net, 1024, 3, 3, 1, 1, 1, 1, c22_bin, true);
    tk::dnn::Activation a22(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c23(&net, 1024, 3, 3, 1, 1, 1, 1, c23_bin, true);
    tk::dnn::Activation a23(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     c24(&net, 1024, 3, 3, 1, 1, 1, 1, c24_bin, true);
    tk::dnn::Activation a24(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Layer *m25_layers[1] = { &a16 };
    tk::dnn::Route      m25(&net, m25_layers, 1);
    tk::dnn::Conv2d     c26(&net, 64, 1, 1, 1, 1, 0, 0,   c26_bin, true);
    tk::dnn::Activation a26(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Reorg      r27(&net, 2);

    tk::dnn::Layer *m28_layers[2] = { &r27, &a24 };
    tk::dnn::Route      m28(&net, m28_layers, 2);

    tk::dnn::Conv2d     c29(&net, 1024, 3, 3, 1, 1, 1, 1, c29_bin, true);
    tk::dnn::Activation a29(&net, CUDNN_ACTIVATION_RELU);   
    tk::dnn::Conv2d     c30(&net, 425, 1, 1, 1, 1, 0, 0,  c30_bin, false);
    tk::dnn::Region     g31(&net, 80, 4, 5);
    
    tk::dnn::RegionInterpret rI(dim, g31.output_dim, 80, 4, 5, 0.3f, g31_bin);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    
    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, "yolo_relu.rt");

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
    std::cout<<"CUDNN vs correct"; checkResult(out_dim, out_data, out);
    std::cout<<"TRT   vs correct"; checkResult(out_dim, out_data2, out);
    std::cout<<"CUDNN vs TRT    "; checkResult(out_dim, out_data, out_data2);
  
    std::cout<<"\n\nDetected objects: \n";
    dnnType *output_h = new dnnType[rI.output_dim.tot()];
    checkCuda(cudaMemcpy(output_h, out_data2, 
              rI.output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost));
    rI.interpretData(output_h, 608, 608);
    rI.showImageResult(input_h);

    return 0;
}
