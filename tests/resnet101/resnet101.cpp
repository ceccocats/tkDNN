#include <iostream>
#include "tkdnn.h"

const char *input_bin = "../tests/resnet101/debug/input.bin";
const char *conv1_bin = "../tests/resnet101/layers/conv1.bin";

//layer1
const char *layer1_0_conv1_bin = "../tests/resnet101/layers/layer1-0-conv1.bin";
const char *layer1_0_conv2_bin = "../tests/resnet101/layers/layer1-0-conv2.bin";
const char *layer1_0_conv3_bin = "../tests/resnet101/layers/layer1-0-conv3.bin";
const char *layer1_0_downsample_0_bin = "../tests/resnet101/layers/layer1-0-downsample-0.bin";

const char *layer1_1_conv1_bin = "../tests/resnet101/layers/layer1-1-conv1.bin";
const char *layer1_1_conv2_bin = "../tests/resnet101/layers/layer1-1-conv2.bin";
const char *layer1_1_conv3_bin = "../tests/resnet101/layers/layer1-1-conv3.bin";

const char *layer1_2_conv1_bin = "../tests/resnet101/layers/layer1-2-conv1.bin";
const char *layer1_2_conv2_bin = "../tests/resnet101/layers/layer1-2-conv2.bin";
const char *layer1_2_conv3_bin = "../tests/resnet101/layers/layer1-2-conv3.bin";

//layer2
const char *layer2_0_conv1_bin = "../tests/resnet101/layers/layer2-0-conv1.bin";
const char *layer2_0_conv2_bin = "../tests/resnet101/layers/layer2-0-conv2.bin";
const char *layer2_0_conv3_bin = "../tests/resnet101/layers/layer2-0-conv3.bin";
const char *layer2_0_downsample_0_bin = "../tests/resnet101/layers/layer2-0-downsample-0.bin";

const char *layer2_1_conv1_bin = "../tests/resnet101/layers/layer2-1-conv1.bin";
const char *layer2_1_conv2_bin = "../tests/resnet101/layers/layer2-1-conv2.bin";
const char *layer2_1_conv3_bin = "../tests/resnet101/layers/layer2-1-conv3.bin";

const char *layer2_2_conv1_bin = "../tests/resnet101/layers/layer2-2-conv1.bin";
const char *layer2_2_conv2_bin = "../tests/resnet101/layers/layer2-2-conv2.bin";
const char *layer2_2_conv3_bin = "../tests/resnet101/layers/layer2-2-conv3.bin";

const char *layer2_3_conv1_bin = "../tests/resnet101/layers/layer2-3-conv1.bin";
const char *layer2_3_conv2_bin = "../tests/resnet101/layers/layer2-3-conv2.bin";
const char *layer2_3_conv3_bin = "../tests/resnet101/layers/layer2-3-conv3.bin";

//layer3
const char *layer3_0_conv1_bin = "../tests/resnet101/layers/layer3-0-conv1.bin";
const char *layer3_0_conv2_bin = "../tests/resnet101/layers/layer3-0-conv2.bin";
const char *layer3_0_conv3_bin = "../tests/resnet101/layers/layer3-0-conv3.bin";
const char *layer3_0_downsample_0_bin = "../tests/resnet101/layers/layer3-0-downsample-0.bin";

const char *layer3_1_conv1_bin = "../tests/resnet101/layers/layer3-1-conv1.bin";
const char *layer3_1_conv2_bin = "../tests/resnet101/layers/layer3-1-conv2.bin";
const char *layer3_1_conv3_bin = "../tests/resnet101/layers/layer3-1-conv3.bin";

const char *layer3_2_conv1_bin = "../tests/resnet101/layers/layer3-2-conv1.bin";
const char *layer3_2_conv2_bin = "../tests/resnet101/layers/layer3-2-conv2.bin";
const char *layer3_2_conv3_bin = "../tests/resnet101/layers/layer3-2-conv3.bin";

const char *layer3_3_conv1_bin = "../tests/resnet101/layers/layer3-3-conv1.bin";
const char *layer3_3_conv2_bin = "../tests/resnet101/layers/layer3-3-conv2.bin";
const char *layer3_3_conv3_bin = "../tests/resnet101/layers/layer3-3-conv3.bin";

const char *layer3_4_conv1_bin = "../tests/resnet101/layers/layer3-4-conv1.bin";
const char *layer3_4_conv2_bin = "../tests/resnet101/layers/layer3-4-conv2.bin";
const char *layer3_4_conv3_bin = "../tests/resnet101/layers/layer3-4-conv3.bin";

const char *layer3_5_conv1_bin = "../tests/resnet101/layers/layer3-5-conv1.bin";
const char *layer3_5_conv2_bin = "../tests/resnet101/layers/layer3-5-conv2.bin";
const char *layer3_5_conv3_bin = "../tests/resnet101/layers/layer3-5-conv3.bin";

const char *layer3_6_conv1_bin = "../tests/resnet101/layers/layer3-6-conv1.bin";
const char *layer3_6_conv2_bin = "../tests/resnet101/layers/layer3-6-conv2.bin";
const char *layer3_6_conv3_bin = "../tests/resnet101/layers/layer3-6-conv3.bin";

const char *layer3_7_conv1_bin = "../tests/resnet101/layers/layer3-7-conv1.bin";
const char *layer3_7_conv2_bin = "../tests/resnet101/layers/layer3-7-conv2.bin";
const char *layer3_7_conv3_bin = "../tests/resnet101/layers/layer3-7-conv3.bin";

const char *layer3_8_conv1_bin = "../tests/resnet101/layers/layer3-8-conv1.bin";
const char *layer3_8_conv2_bin = "../tests/resnet101/layers/layer3-8-conv2.bin";
const char *layer3_8_conv3_bin = "../tests/resnet101/layers/layer3-8-conv3.bin";

const char *layer3_9_conv1_bin = "../tests/resnet101/layers/layer3-9-conv1.bin";
const char *layer3_9_conv2_bin = "../tests/resnet101/layers/layer3-9-conv2.bin";
const char *layer3_9_conv3_bin = "../tests/resnet101/layers/layer3-9-conv3.bin";

const char *layer3_10_conv1_bin = "../tests/resnet101/layers/layer3-10-conv1.bin";
const char *layer3_10_conv2_bin = "../tests/resnet101/layers/layer3-10-conv2.bin";
const char *layer3_10_conv3_bin = "../tests/resnet101/layers/layer3-10-conv3.bin";

const char *layer3_11_conv1_bin = "../tests/resnet101/layers/layer3-11-conv1.bin";
const char *layer3_11_conv2_bin = "../tests/resnet101/layers/layer3-11-conv2.bin";
const char *layer3_11_conv3_bin = "../tests/resnet101/layers/layer3-11-conv3.bin";

const char *layer3_12_conv1_bin = "../tests/resnet101/layers/layer3-12-conv1.bin";
const char *layer3_12_conv2_bin = "../tests/resnet101/layers/layer3-12-conv2.bin";
const char *layer3_12_conv3_bin = "../tests/resnet101/layers/layer3-12-conv3.bin";

const char *layer3_13_conv1_bin = "../tests/resnet101/layers/layer3-13-conv1.bin";
const char *layer3_13_conv2_bin = "../tests/resnet101/layers/layer3-13-conv2.bin";
const char *layer3_13_conv3_bin = "../tests/resnet101/layers/layer3-13-conv3.bin";

const char *layer3_14_conv1_bin = "../tests/resnet101/layers/layer3-14-conv1.bin";
const char *layer3_14_conv2_bin = "../tests/resnet101/layers/layer3-14-conv2.bin";
const char *layer3_14_conv3_bin = "../tests/resnet101/layers/layer3-14-conv3.bin";

const char *layer3_15_conv1_bin = "../tests/resnet101/layers/layer3-15-conv1.bin";
const char *layer3_15_conv2_bin = "../tests/resnet101/layers/layer3-15-conv2.bin";
const char *layer3_15_conv3_bin = "../tests/resnet101/layers/layer3-15-conv3.bin";

const char *layer3_16_conv1_bin = "../tests/resnet101/layers/layer3-16-conv1.bin";
const char *layer3_16_conv2_bin = "../tests/resnet101/layers/layer3-16-conv2.bin";
const char *layer3_16_conv3_bin = "../tests/resnet101/layers/layer3-16-conv3.bin";

const char *layer3_17_conv1_bin = "../tests/resnet101/layers/layer3-17-conv1.bin";
const char *layer3_17_conv2_bin = "../tests/resnet101/layers/layer3-17-conv2.bin";
const char *layer3_17_conv3_bin = "../tests/resnet101/layers/layer3-17-conv3.bin";

const char *layer3_18_conv1_bin = "../tests/resnet101/layers/layer3-18-conv1.bin";
const char *layer3_18_conv2_bin = "../tests/resnet101/layers/layer3-18-conv2.bin";
const char *layer3_18_conv3_bin = "../tests/resnet101/layers/layer3-18-conv3.bin";

const char *layer3_19_conv1_bin = "../tests/resnet101/layers/layer3-19-conv1.bin";
const char *layer3_19_conv2_bin = "../tests/resnet101/layers/layer3-19-conv2.bin";
const char *layer3_19_conv3_bin = "../tests/resnet101/layers/layer3-19-conv3.bin";

const char *layer3_20_conv1_bin = "../tests/resnet101/layers/layer3-20-conv1.bin";
const char *layer3_20_conv2_bin = "../tests/resnet101/layers/layer3-20-conv2.bin";
const char *layer3_20_conv3_bin = "../tests/resnet101/layers/layer3-20-conv3.bin";

const char *layer3_21_conv1_bin = "../tests/resnet101/layers/layer3-21-conv1.bin";
const char *layer3_21_conv2_bin = "../tests/resnet101/layers/layer3-21-conv2.bin";
const char *layer3_21_conv3_bin = "../tests/resnet101/layers/layer3-21-conv3.bin";

const char *layer3_22_conv1_bin = "../tests/resnet101/layers/layer3-22-conv1.bin";
const char *layer3_22_conv2_bin = "../tests/resnet101/layers/layer3-22-conv2.bin";
const char *layer3_22_conv3_bin = "../tests/resnet101/layers/layer3-22-conv3.bin";


//layer4
const char *layer4_0_conv1_bin = "../tests/resnet101/layers/layer4-0-conv1.bin";
const char *layer4_0_conv2_bin = "../tests/resnet101/layers/layer4-0-conv2.bin";
const char *layer4_0_conv3_bin = "../tests/resnet101/layers/layer4-0-conv3.bin";
const char *layer4_0_downsample_0_bin = "../tests/resnet101/layers/layer4-0-downsample-0.bin";

const char *layer4_1_conv1_bin = "../tests/resnet101/layers/layer4-1-conv1.bin";
const char *layer4_1_conv2_bin = "../tests/resnet101/layers/layer4-1-conv2.bin";
const char *layer4_1_conv3_bin = "../tests/resnet101/layers/layer4-1-conv3.bin";

const char *layer4_2_conv1_bin = "../tests/resnet101/layers/layer4-2-conv1.bin";
const char *layer4_2_conv2_bin = "../tests/resnet101/layers/layer4-2-conv2.bin";
const char *layer4_2_conv3_bin = "../tests/resnet101/layers/layer4-2-conv3.bin";

//final
const char *fc_bin = "../tests/resnet101/layers/fc.bin";

const char *output_bin = "../tests/resnet101/debug/layer1-0-relu.bin";

int main()
{

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 224, 224, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d conv1(&net, 64, 7, 7, 2, 2, 3, 3, conv1_bin, true);
    tk::dnn::Activation relu3(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Pooling maxpool4(&net, 3, 3, 2, 2, 1, 1, tk::dnn::POOLING_MAX);

    //layer 1
    tk::dnn::Conv2d layer1_0_conv1(&net, 64, 1, 1, 1, 1, 0, 0, layer1_0_conv1_bin, true);
    tk::dnn::Activation relu1_0_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d layer1_0_conv2(&net, 64, 3, 3, 1, 1, 1, 1, layer1_0_conv2_bin, true);
    tk::dnn::Activation relu1_0_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d layer1_0_conv3(&net, 256, 1, 1, 1, 1, 0, 0, layer1_0_conv3_bin, true);
    
    tk::dnn::Layer *m83_layers[1] = { &maxpool4 };
    tk::dnn::Route      m83  (&net, m83_layers, 1);
    tk::dnn::Conv2d layer1_0_downsample_0(&net, 256, 1, 1, 1, 1, 0, 0, layer1_0_downsample_0_bin, true);
    
    tk::dnn::Shortcut   s1_0 (&net, &layer1_0_conv3);
    tk::dnn::Activation layer1_0_relu(&net, CUDNN_ACTIVATION_RELU);
/*
    
    tk::dnn::Conv2d layer1_1_conv1(&net, 64, 1, 1, 1, 1, 1, 1, layer1_1_conv1_bin, true);
    tk::dnn::Conv2d layer1_1_conv2(&net, 64, 3, 3, 1, 1, 1, 1, layer1_1_conv2_bin, true);
    tk::dnn::Conv2d layer1_1_conv3(&net, 256, 1, 1, 1, 1, 1, 1, layer1_1_conv3_bin, true);
    tk::dnn::Activation layer1_1_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer1_2_conv1(&net, 64, 1, 1, 1, 1, 1, 1, layer1_2_conv1_bin, true);
    tk::dnn::Conv2d layer1_2_conv2(&net, 64, 3, 3, 1, 1, 1, 1, layer1_2_conv2_bin, true);
    tk::dnn::Conv2d layer1_2_conv3(&net, 256, 1, 1, 1, 1, 1, 1, layer1_2_conv3_bin, true);
    tk::dnn::Activation layer1_2_relu(&net, CUDNN_ACTIVATION_RELU);

    
    //layer 2
    tk::dnn::Conv2d layer2_0_conv1(&net, 128, 1, 1, 1, 1, 1, 1, layer2_0_conv1_bin, true);
    tk::dnn::Conv2d layer2_0_conv2(&net, 128, 3, 3, 2, 2, 1, 1, layer2_0_conv2_bin, true);
    tk::dnn::Conv2d layer2_0_conv3(&net, 512, 1, 1, 1, 1, 1, 1, layer2_0_conv3_bin, true);
    tk::dnn::Activation layer2_0_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d layer2_0_downsample_0(&net, 512, 1, 1, 2, 2, 1, 1, layer2_0_downsample_0, true);

    tk::dnn::Conv2d layer2_1_conv1(&net, 128, 1, 1, 1, 1, 1, 1, layer2_1_conv1_bin, true);
    tk::dnn::Conv2d layer2_1_conv2(&net, 128, 3, 3, 1, 1, 1, 1, layer2_1_conv2_bin, true);
    tk::dnn::Conv2d layer2_1_conv3(&net, 512, 1, 1, 1, 1, 1, 1, layer2_1_conv3_bin, true);
    tk::dnn::Activation layer2_1_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer2_2_conv1(&net, 128, 1, 1, 1, 1, 1, 1, layer2_2_conv1_bin, true);
    tk::dnn::Conv2d layer2_2_conv2(&net, 128, 3, 3, 1, 1, 1, 1, layer2_2_conv2_bin, true);
    tk::dnn::Conv2d layer2_2_conv3(&net, 512, 1, 1, 1, 1, 1, 1, layer2_2_conv3_bin, true);
    tk::dnn::Activation layer2_2_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer2_3_conv1(&net, 128, 1, 1, 1, 1, 1, 1, layer2_3_conv1_bin, true);
    tk::dnn::Conv2d layer2_3_conv2(&net, 128, 3, 3, 1, 1, 1, 1, layer2_3_conv2_bin, true);
    tk::dnn::Conv2d layer2_3_conv3(&net, 512, 1, 1, 1, 1, 1, 1, layer2_3_conv3_bin, true);
    tk::dnn::Activation layer2_3_relu(&net, CUDNN_ACTIVATION_RELU);

    //layer 3
    tk::dnn::Conv2d layer3_0_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_0_conv1_bin, true);
    tk::dnn::Conv2d layer3_0_conv2(&net, 256, 3, 3, 2, 2, 1, 1, layer3_0_conv2_bin, true);
    tk::dnn::Conv2d layer3_0_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_0_conv3_bin, true);
    tk::dnn::Activation layer3_0_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d layer3_0_downsample_0(&net, 1024, 1, 1, 2, 2, 1, 1, layer3_0_downsample_0, true);

    tk::dnn::Conv2d layer3_1_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_1_conv1_bin, true);
    tk::dnn::Conv2d layer3_1_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_1_conv2_bin, true);
    tk::dnn::Conv2d layer3_1_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_1_conv3_bin, true);
    tk::dnn::Activation layer3_1_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_2_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_2_conv1_bin, true);
    tk::dnn::Conv2d layer3_2_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_2_conv2_bin, true);
    tk::dnn::Conv2d layer3_2_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_2_conv3_bin, true);
    tk::dnn::Activation layer3_2_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_3_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_3_conv1_bin, true);
    tk::dnn::Conv2d layer3_3_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_3_conv2_bin, true);
    tk::dnn::Conv2d layer3_3_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_3_conv3_bin, true);
    tk::dnn::Activation layer3_3_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_4_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_4_conv1_bin, true);
    tk::dnn::Conv2d layer3_4_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_4_conv2_bin, true);
    tk::dnn::Conv2d layer3_4_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_4_conv3_bin, true);
    tk::dnn::Activation layer3_4_relu(&net, CUDNN_ACTIVATION_RELU);
    
    tk::dnn::Conv2d layer3_5_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_5_conv1_bin, true);
    tk::dnn::Conv2d layer3_5_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_5_conv2_bin, true);
    tk::dnn::Conv2d layer3_5_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_5_conv3_bin, true);
    tk::dnn::Activation layer3_5_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_6_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_6_conv1_bin, true);
    tk::dnn::Conv2d layer3_6_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_6_conv2_bin, true);
    tk::dnn::Conv2d layer3_6_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_6_conv3_bin, true);
    tk::dnn::Activation layer3_6_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_7_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_7_conv1_bin, true);
    tk::dnn::Conv2d layer3_7_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_7_conv2_bin, true);
    tk::dnn::Conv2d layer3_7_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_7_conv3_bin, true);
    tk::dnn::Activation layer3_7_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_8_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_8_conv1_bin, true);
    tk::dnn::Conv2d layer3_8_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_8_conv2_bin, true);
    tk::dnn::Conv2d layer3_8_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_8_conv3_bin, true);
    tk::dnn::Activation layer3_8_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_9_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_9_conv1_bin, true);
    tk::dnn::Conv2d layer3_9_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_9_conv2_bin, true);
    tk::dnn::Conv2d layer3_9_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_9_conv3_bin, true);
    tk::dnn::Activation layer3_9_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_10_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_10_conv1_bin, true);
    tk::dnn::Conv2d layer3_10_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_10_conv2_bin, true);
    tk::dnn::Conv2d layer3_10_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_10_conv3_bin, true);
    tk::dnn::Activation layer3_10_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_11_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_11_conv1_bin, true);
    tk::dnn::Conv2d layer3_11_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_11_conv2_bin, true);
    tk::dnn::Conv2d layer3_11_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_11_conv3_bin, true);
    tk::dnn::Activation layer3_11_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_12_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_12_conv1_bin, true);
    tk::dnn::Conv2d layer3_12_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_12_conv2_bin, true);
    tk::dnn::Conv2d layer3_12_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_12_conv3_bin, true);
    tk::dnn::Activation layer3_12_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_13_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_13_conv1_bin, true);
    tk::dnn::Conv2d layer3_13_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_13_conv2_bin, true);
    tk::dnn::Conv2d layer3_13_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_13_conv3_bin, true);
    tk::dnn::Activation layer3_13_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_14_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_14_conv1_bin, true);
    tk::dnn::Conv2d layer3_14_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_14_conv2_bin, true);
    tk::dnn::Conv2d layer3_14_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_14_conv3_bin, true);
    tk::dnn::Activation layer3_14_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_15_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_15_conv1_bin, true);
    tk::dnn::Conv2d layer3_15_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_15_conv2_bin, true);
    tk::dnn::Conv2d layer3_15_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_15_conv3_bin, true);
    tk::dnn::Activation layer3_15_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_16_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_16_conv1_bin, true);
    tk::dnn::Conv2d layer3_16_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_16_conv2_bin, true);
    tk::dnn::Conv2d layer3_16_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_16_conv3_bin, true);
    tk::dnn::Activation layer3_16_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_17_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_17_conv1_bin, true);
    tk::dnn::Conv2d layer3_17_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_17_conv2_bin, true);
    tk::dnn::Conv2d layer3_17_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_17_conv3_bin, true);
    tk::dnn::Activation layer3_17_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_18_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_18_conv1_bin, true);
    tk::dnn::Conv2d layer3_18_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_18_conv2_bin, true);
    tk::dnn::Conv2d layer3_18_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_18_conv3_bin, true);
    tk::dnn::Activation layer3_18_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_19_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_19_conv1_bin, true);
    tk::dnn::Conv2d layer3_19_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_19_conv2_bin, true);
    tk::dnn::Conv2d layer3_19_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_19_conv3_bin, true);
    tk::dnn::Activation layer3_19_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_20_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_20_conv1_bin, true);
    tk::dnn::Conv2d layer3_20_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_20_conv2_bin, true);
    tk::dnn::Conv2d layer3_20_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_20_conv3_bin, true);
    tk::dnn::Activation layer3_20_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_21_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_21_conv1_bin, true);
    tk::dnn::Conv2d layer3_21_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_21_conv2_bin, true);
    tk::dnn::Conv2d layer3_21_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_21_conv3_bin, true);
    tk::dnn::Activation layer3_21_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer3_22_conv1(&net, 256, 1, 1, 1, 1, 1, 1, layer3_22_conv1_bin, true);
    tk::dnn::Conv2d layer3_22_conv2(&net, 256, 3, 3, 1, 1, 1, 1, layer3_22_conv2_bin, true);
    tk::dnn::Conv2d layer3_22_conv3(&net, 1024, 1, 1, 1, 1, 1, 1, layer3_22_conv3_bin, true);
    tk::dnn::Activation layer3_22_relu(&net, CUDNN_ACTIVATION_RELU);

    //layer 4
    tk::dnn::Conv2d layer4_0_conv1(&net, 512, 1, 1, 1, 1, 1, 1, layer4_0_conv1_bin, true);
    tk::dnn::Conv2d layer4_0_conv2(&net, 512, 3, 3, 2, 2, 1, 1, layer4_0_conv2_bin, true);
    tk::dnn::Conv2d layer4_0_conv3(&net, 2048, 1, 1, 1, 1, 1, 1, layer4_0_conv3_bin, true);
    tk::dnn::Activation layer4_0_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d layer4_0_downsample_0(&net, 2048, 1, 1, 2, 2, 1, 1, layer4_0_downsample_0, true);

    tk::dnn::Conv2d layer4_1_conv1(&net, 512, 1, 1, 1, 1, 1, 1, layer4_1_conv1_bin, true);
    tk::dnn::Conv2d layer4_1_conv2(&net, 512, 3, 3, 1, 1, 1, 1, layer4_1_conv2_bin, true);
    tk::dnn::Conv2d layer4_1_conv3(&net, 2048, 1, 1, 1, 1, 1, 1, layer4_1_conv3_bin, true);
    tk::dnn::Activation layer4_1_relu(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d layer4_2_conv1(&net, 512, 1, 1, 1, 1, 1, 1, layer4_2_conv1_bin, true);
    tk::dnn::Conv2d layer4_2_conv2(&net, 512, 3, 3, 1, 1, 1, 1, layer4_2_conv2_bin, true);
    tk::dnn::Conv2d layer4_2_conv3(&net, 2048, 1, 1, 1, 1, 1, 1, layer4_2_conv3_bin, true);
    tk::dnn::Activation layer4_2_relu(&net, CUDNN_ACTIVATION_RELU);


    //final
    tk::dnn::Pooling avgpool(&net, 3, 3, 2, 2, tk::dnn::POOLING_AVERAGE);
    tk::dnn::Dense   fc(&net, 1000, fc_bin);
*/
    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    printDeviceVector(64, data, true);

    //print network model
    net.print();
/*
    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, "resnet101.rt");
*/
    
    tk::dnn::dataDim_t out_dim;
    out_dim = net.layers[net.num_layers-1]->output_dim;
    dnnType *cudnn_out, *rt_out;

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TIMER_START
        net.infer(dim1, data);
        TIMER_STOP
        dim1.print();
    }
    cudnn_out = net.layers[net.num_layers-1]->dstData;

    printDeviceVector(64, cudnn_out, true);
/*
    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TIMER_START
        netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }
    rt_out = (dnnType *)netRT.buffersRT[1];
*/

    printCenteredTitle(std::string(" RESNET CHECK RESULTS ").c_str(), '=', 30);
    dnnType *out, *out_h;
    int odim = out_dim.tot();
    readBinaryFile(output_bin, odim, &out_h, &out);
    std::cout << "CUDNN vs correct";
    checkResult(odim, cudnn_out, out);
/*
    std::cout << "TRT   vs correct";
    checkResult(odim, rt_out, out);
    std::cout << "CUDNN vs TRT    ";
    checkResult(odim, cudnn_out, rt_out);
*/
    return 0;
}
