#include <iostream>
#include "tkdnn.h"


const char *output_bin1 = "mobilenetv2ssd/debug/classification_headers-5.bin";
const char *output_bin2 = "mobilenetv2ssd/debug/regression_headers-5.bin";
const char *input_bin = "mobilenetv2ssd/debug/input.bin";

const char *conv0_bin = "mobilenetv2ssd/layers/base_net-0-0.bin";
const char *inverted_residual1[] = {
    "mobilenetv2ssd/layers/base_net-1-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-1-conv-3.bin"};
const char *inverted_residual2[] = {
    "mobilenetv2ssd/layers/base_net-2-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-2-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-2-conv-6.bin"};
const char *inverted_residual3[] = {
    "mobilenetv2ssd/layers/base_net-3-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-3-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-3-conv-6.bin"};
const char *inverted_residual4[] = {
    "mobilenetv2ssd/layers/base_net-4-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-4-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-4-conv-6.bin"};
const char *inverted_residual5[] = {
    "mobilenetv2ssd/layers/base_net-5-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-5-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-5-conv-6.bin"};
const char *inverted_residual6[] = {
    "mobilenetv2ssd/layers/base_net-6-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-6-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-6-conv-6.bin"};
const char *inverted_residual7[] = {
    "mobilenetv2ssd/layers/base_net-7-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-7-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-7-conv-6.bin"};
const char *inverted_residual8[] = {
    "mobilenetv2ssd/layers/base_net-8-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-8-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-8-conv-6.bin"};
const char *inverted_residual9[] = {
    "mobilenetv2ssd/layers/base_net-9-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-9-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-9-conv-6.bin"};
const char *inverted_residual10[] = {
    "mobilenetv2ssd/layers/base_net-10-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-10-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-10-conv-6.bin"};
const char *inverted_residual11[] = {
    "mobilenetv2ssd/layers/base_net-11-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-11-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-11-conv-6.bin"};
const char *inverted_residual12[] = {
    "mobilenetv2ssd/layers/base_net-12-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-12-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-12-conv-6.bin"};
const char *inverted_residual13[] = {
    "mobilenetv2ssd/layers/base_net-13-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-13-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-13-conv-6.bin"};
const char *inverted_residual14[] = {
    "mobilenetv2ssd/layers/base_net-14-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-14-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-14-conv-6.bin"};
const char *inverted_residual15[] = {
    "mobilenetv2ssd/layers/base_net-15-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-15-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-15-conv-6.bin"};
const char *inverted_residual16[] = {
    "mobilenetv2ssd/layers/base_net-16-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-16-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-16-conv-6.bin"};
const char *inverted_residual17[] = {
    "mobilenetv2ssd/layers/base_net-17-conv-0.bin",
    "mobilenetv2ssd/layers/base_net-17-conv-3.bin",
    "mobilenetv2ssd/layers/base_net-17-conv-6.bin"};

const char *conv18 = "mobilenetv2ssd/layers/base_net-18-0.bin";

const char *extras0[] = {
    "mobilenetv2ssd/layers/extras-0-conv-0.bin",
    "mobilenetv2ssd/layers/extras-0-conv-3.bin",
    "mobilenetv2ssd/layers/extras-0-conv-6.bin"};
const char *extras1[] = {
    "mobilenetv2ssd/layers/extras-1-conv-0.bin",
    "mobilenetv2ssd/layers/extras-1-conv-3.bin",
    "mobilenetv2ssd/layers/extras-1-conv-6.bin"};
const char *extras2[] = {
    "mobilenetv2ssd/layers/extras-2-conv-0.bin",
    "mobilenetv2ssd/layers/extras-2-conv-3.bin",
    "mobilenetv2ssd/layers/extras-2-conv-6.bin"};
const char *extras3[] = {
    "mobilenetv2ssd/layers/extras-3-conv-0.bin",
    "mobilenetv2ssd/layers/extras-3-conv-3.bin",
    "mobilenetv2ssd/layers/extras-3-conv-6.bin"};

const char *classification_header0[] = {
    "mobilenetv2ssd/layers/classification_headers-0-0.bin",
    "mobilenetv2ssd/layers/classification_headers-0-3.bin"};
const char *classification_header1[] = {
    "mobilenetv2ssd/layers/classification_headers-1-0.bin",
    "mobilenetv2ssd/layers/classification_headers-1-3.bin"};
const char *classification_header2[] = {
    "mobilenetv2ssd/layers/classification_headers-2-0.bin",
    "mobilenetv2ssd/layers/classification_headers-2-3.bin"};
const char *classification_header3[] = {
    "mobilenetv2ssd/layers/classification_headers-3-0.bin",
    "mobilenetv2ssd/layers/classification_headers-3-3.bin"};
const char *classification_header4[] = {
    "mobilenetv2ssd/layers/classification_headers-4-0.bin",
    "mobilenetv2ssd/layers/classification_headers-4-3.bin"};

const char *classification_header5 = "mobilenetv2ssd/layers/classification_headers-5.bin";

const char *regression_header0[] = {
    "mobilenetv2ssd/layers/regression_headers-0-0.bin",
    "mobilenetv2ssd/layers/regression_headers-0-3.bin"};
const char *regression_header1[] = {
    "mobilenetv2ssd/layers/regression_headers-1-0.bin",
    "mobilenetv2ssd/layers/regression_headers-1-3.bin"};
const char *regression_header2[] = {
    "mobilenetv2ssd/layers/regression_headers-2-0.bin",
    "mobilenetv2ssd/layers/regression_headers-2-3.bin"};
const char *regression_header3[] = {
    "mobilenetv2ssd/layers/regression_headers-3-0.bin",
    "mobilenetv2ssd/layers/regression_headers-3-3.bin"};
const char *regression_header4[] = {
    "mobilenetv2ssd/layers/regression_headers-4-0.bin",
    "mobilenetv2ssd/layers/regression_headers-4-3.bin"};

const char *regression_header5 = "mobilenetv2ssd/layers/regression_headers-5.bin";


int main()
{

    downloadWeightsifDoNotExist(input_bin, "mobilenetv2ssd", "https://cloud.hipert.unimore.it/s/x4ZfxBKN23zAJQp/download");

    int classes = 21;

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 300, 300, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d conv1(&net, 32, 3, 3, 2, 2, 1, 1, conv0_bin, true);
    tk::dnn::Activation relu3(&net, CUDNN_ACTIVATION_RELU);

    //Inverted Residual 1

    tk::dnn::Conv2d conv2(&net, 32, 3, 3, 1, 1, 1, 1, inverted_residual1[0], true, false, 32);
    tk::dnn::Activation relu5(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d conv3(&net, 16, 1, 1, 1, 1, 0, 0, inverted_residual1[1], true);

    //Inverted Residual 2
    tk::dnn::Conv2d ir_2_conv1(&net, 96, 1, 1, 1, 1, 0, 0, inverted_residual2[0], true);
    tk::dnn::Activation relu_2_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_2_conv2(&net, 96, 3, 3, 2, 2, 1, 1, inverted_residual2[1], true, false, 96);
    tk::dnn::Activation relu_2_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_2_conv3(&net, 24, 1, 1, 1, 1, 0, 0, inverted_residual2[2], true);

    //Inverted Residual 3
    tk::dnn::Layer *last = &ir_2_conv3;
    tk::dnn::Conv2d ir_3_conv1(&net, 144, 1, 1, 1, 1, 0, 0, inverted_residual3[0], true);
    tk::dnn::Activation relu_3_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_3_conv2(&net, 144, 3, 3, 1, 1, 1, 1, inverted_residual3[1], true, false, 144);
    tk::dnn::Activation relu_3_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_3_conv3(&net, 24, 1, 1, 1, 1, 0, 0, inverted_residual3[2], true);

    tk::dnn::Shortcut s3_0(&net, last);
    // //Inverted Residual 4
    tk::dnn::Conv2d ir_4_conv1(&net, 144, 1, 1, 1, 1, 0, 0, inverted_residual4[0], true);
    tk::dnn::Activation relu_4_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_4_conv2(&net, 144, 3, 3, 2, 2, 1, 1, inverted_residual4[1], true, false, 144);
    tk::dnn::Activation relu_4_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_4_conv3(&net, 32, 1, 1, 1, 1, 0, 0, inverted_residual4[2], true);

    // // //Inverted Residual 5
    last = &ir_4_conv3;
    tk::dnn::Conv2d ir_5_conv1(&net, 192, 1, 1, 1, 1, 0, 0, inverted_residual5[0], true);
    tk::dnn::Activation relu_5_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_5_conv2(&net, 192, 3, 3, 1, 1, 1, 1, inverted_residual5[1], true, false, 192);
    tk::dnn::Activation relu_5_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_5_conv3(&net, 32, 1, 1, 1, 1, 0, 0, inverted_residual5[2], true);

    tk::dnn::Shortcut s5_0(&net, last);
    // // // //Inverted Residual 6
    last = &s5_0;
    tk::dnn::Conv2d ir_6_conv1(&net, 192, 1, 1, 1, 1, 0, 0, inverted_residual6[0], true);
    tk::dnn::Activation relu_6_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_6_conv2(&net, 192, 3, 3, 1, 1, 1, 1, inverted_residual6[1], true, false, 192);
    tk::dnn::Activation relu_6_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_6_conv3(&net, 32, 1, 1, 1, 1, 0, 0, inverted_residual6[2], true);

    tk::dnn::Shortcut s6_0(&net, last);
    //Inverted Residual 7
    tk::dnn::Conv2d ir_7_conv1(&net, 192, 1, 1, 1, 1, 0, 0, inverted_residual7[0], true);
    tk::dnn::Activation relu_7_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_7_conv2(&net, 192, 3, 3, 2, 2, 1, 1, inverted_residual7[1], true, false, 192);
    tk::dnn::Activation relu_7_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_7_conv3(&net, 64, 1, 1, 1, 1, 0, 0, inverted_residual7[2], true);

    // //Inverted Residual 8
    last = &ir_7_conv3;
    tk::dnn::Conv2d ir_8_conv1(&net, 384, 1, 1, 1, 1, 0, 0, inverted_residual8[0], true);
    tk::dnn::Activation relu_8_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_8_conv2(&net, 384, 3, 3, 1, 1, 1, 1, inverted_residual8[1], true, false, 384);
    tk::dnn::Activation relu_8_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_8_conv3(&net, 64, 1, 1, 1, 1, 0, 0, inverted_residual8[2], true);

    tk::dnn::Shortcut s8_0(&net, last);
    //Inverted Residual 9
    last = &s8_0;
    tk::dnn::Conv2d ir_9_conv1(&net, 384, 1, 1, 1, 1, 0, 0, inverted_residual9[0], true);
    tk::dnn::Activation relu_9_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_9_conv2(&net, 384, 3, 3, 1, 1, 1, 1, inverted_residual9[1], true, false, 384);
    tk::dnn::Activation relu_9_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_9_conv3(&net, 64, 1, 1, 1, 1, 0, 0, inverted_residual9[2], true);

    tk::dnn::Shortcut s9_0(&net, last);
    //Inverted Residual 10
    last = &s9_0;
    tk::dnn::Conv2d ir_10_conv1(&net, 384, 1, 1, 1, 1, 0, 0, inverted_residual10[0], true);
    tk::dnn::Activation relu_10_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_10_conv2(&net, 384, 3, 3, 1, 1, 1, 1, inverted_residual10[1], true, false, 384);
    tk::dnn::Activation relu_10_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_10_conv3(&net, 64, 1, 1, 1, 1, 0, 0, inverted_residual10[2], true);

    tk::dnn::Shortcut s10_0(&net, last);
    //Inverted Residual 11
    tk::dnn::Conv2d ir_11_conv1(&net, 384, 1, 1, 1, 1, 0, 0, inverted_residual11[0], true);
    tk::dnn::Activation relu_11_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_11_conv2(&net, 384, 3, 3, 1, 1, 1, 1, inverted_residual11[1], true, false, 384);
    tk::dnn::Activation relu_11_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_11_conv3(&net, 96, 1, 1, 1, 1, 0, 0, inverted_residual11[2], true);

    last = &ir_11_conv3;
    //Inverted Residual 12
    tk::dnn::Conv2d ir_12_conv1(&net, 576, 1, 1, 1, 1, 0, 0, inverted_residual12[0], true);
    tk::dnn::Activation relu_12_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_12_conv2(&net, 576, 3, 3, 1, 1, 1, 1, inverted_residual12[1], true, false, 576);
    tk::dnn::Activation relu_12_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_12_conv3(&net, 96, 1, 1, 1, 1, 0, 0, inverted_residual12[2], true);

    tk::dnn::Shortcut s12_0(&net, last);
    last = &s12_0;
    //Inverted Residual 13
    tk::dnn::Conv2d ir_13_conv1(&net, 576, 1, 1, 1, 1, 0, 0, inverted_residual13[0], true);
    tk::dnn::Activation relu_13_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_13_conv2(&net, 576, 3, 3, 1, 1, 1, 1, inverted_residual13[1], true, false, 576);
    tk::dnn::Activation relu_13_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_13_conv3(&net, 96, 1, 1, 1, 1, 0, 0, inverted_residual13[2], true);

    tk::dnn::Shortcut s13_0(&net, last);
    // //Inverted Residual 14
    tk::dnn::Conv2d ir_14_conv1(&net, 576, 1, 1, 1, 1, 0, 0, inverted_residual14[0], true);
    tk::dnn::Activation relu_14_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_14_conv2(&net, 576, 3, 3, 2, 2, 1, 1, inverted_residual14[1], true, false, 576);
    tk::dnn::Activation relu_14_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_14_conv3(&net, 160, 1, 1, 1, 1, 0, 0, inverted_residual14[2], true);

    // //Inverted Residual 15
    last = &ir_14_conv3;
    tk::dnn::Conv2d ir_15_conv1(&net, 960, 1, 1, 1, 1, 0, 0, inverted_residual15[0], true);
    tk::dnn::Activation relu_15_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_15_conv2(&net, 960, 3, 3, 1, 1, 1, 1, inverted_residual15[1], true, false, 960);
    tk::dnn::Activation relu_15_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_15_conv3(&net, 160, 1, 1, 1, 1, 0, 0, inverted_residual15[2], true);

    tk::dnn::Shortcut s15_0(&net, last);
    //Inverted Residual 16
    last = &s15_0;
    tk::dnn::Conv2d ir_16_conv1(&net, 960, 1, 1, 1, 1, 0, 0, inverted_residual16[0], true);
    tk::dnn::Activation relu_16_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_16_conv2(&net, 960, 3, 3, 1, 1, 1, 1, inverted_residual16[1], true, false, 960);
    tk::dnn::Activation relu_16_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_16_conv3(&net, 160, 1, 1, 1, 1, 0, 0, inverted_residual16[2], true);

    tk::dnn::Shortcut s16_0(&net, last);
    //Inverted Residual 17
    tk::dnn::Conv2d ir_17_conv1(&net, 960, 1, 1, 1, 1, 0, 0, inverted_residual17[0], true);
    tk::dnn::Activation relu_17_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_17_conv2(&net, 960, 3, 3, 1, 1, 1, 1, inverted_residual17[1], true, false, 960);
    tk::dnn::Activation relu_17_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d ir_17_conv3(&net, 320, 1, 1, 1, 1, 0, 0, inverted_residual17[2], true);

    //Conv 18
    tk::dnn::Conv2d ir_18_conv1(&net, 1280, 1, 1, 1, 1, 0, 0, conv18, true);
    tk::dnn::Activation relu_18_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer *header_1[1] = {&relu_18_1};

    // //extras Inverted Residual 0
    tk::dnn::Conv2d e_0_conv1(&net, 256, 1, 1, 1, 1, 0, 0, extras0[0], true);
    tk::dnn::Activation e_relu_0_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d e_0_conv2(&net, 256, 3, 3, 2, 2, 1, 1, extras0[1], true, false, 256);
    tk::dnn::Activation e_relu_0_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d e_0_conv3(&net, 512, 1, 1, 1, 1, 0, 0, extras0[2], true);
    tk::dnn::Layer *header_2[1] = {&e_0_conv3};

    // //extras Inverted Residual 1
    tk::dnn::Conv2d e_1_conv1(&net, 128, 1, 1, 1, 1, 0, 0, extras1[0], true);
    tk::dnn::Activation e_relu_1_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d e_1_conv2(&net, 128, 3, 3, 2, 2, 1, 1, extras1[1], true, false, 128);
    tk::dnn::Activation e_relu_1_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d e_1_conv3(&net, 256, 1, 1, 1, 1, 0, 0, extras1[2], true);
    tk::dnn::Layer *header_3[1] = {&e_1_conv3};

    //extras Inverted Residual 2
    tk::dnn::Conv2d e_2_conv1(&net, 128, 1, 1, 1, 1, 0, 0, extras2[0], true);
    tk::dnn::Activation e_relu_2_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d e_2_conv2(&net, 128, 3, 3, 2, 2, 1, 1, extras2[1], true, false, 128);
    tk::dnn::Activation e_relu_2_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d e_2_conv3(&net, 256, 1, 1, 1, 1, 0, 0, extras2[2], true);
    tk::dnn::Layer *header_4[1] = {&e_2_conv3};

    //extras Inverted Residual 3
    tk::dnn::Conv2d e_3_conv1(&net, 64, 1, 1, 1, 1, 0, 0, extras3[0], true);
    tk::dnn::Activation e_relu_3_1(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d e_3_conv2(&net, 64, 3, 3, 2, 2, 1, 1, extras3[1], true, false, 64);
    tk::dnn::Activation e_relu_3_2(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d e_3_conv3(&net, 64, 1, 1, 1, 1, 0, 0, extras3[2], true);
    tk::dnn::Layer *header_5[1] = {&e_3_conv3};

    // classification header 0
    tk::dnn::Layer *header_0[1] = {&relu_14_1};
    tk::dnn::Route rout_ch_0(&net, header_0, 1);
    tk::dnn::Conv2d ch_0_conv1(&net, 576, 3, 3, 1, 1, 1, 1, classification_header0[0], true, false, 576, true);
    tk::dnn::Activation ch_relu_0_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d ch_0_conv2(&net, 126, 1, 1, 1, 1, 0, 0, classification_header0[1], false);
    tk::dnn::Layer *conf0[1] = {&ch_0_conv2};

    // // classification header 1
    tk::dnn::Route rout_ch_1(&net, header_1, 1);
    tk::dnn::Conv2d ch_1_conv1(&net, 1280, 3, 3, 1, 1, 1, 1, classification_header1[0], true, false, 1280, true);
    tk::dnn::Activation ch_relu_1_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d ch_1_conv2(&net, 126, 1, 1, 1, 1, 0, 0, classification_header1[1], false);
    tk::dnn::Layer *conf1[1] = {&ch_1_conv2};

    // //classification header 2
    tk::dnn::Route rout_ch_2(&net, header_2, 1);
    tk::dnn::Conv2d ch_2_conv1(&net, 512, 3, 3, 1, 1, 1, 1, classification_header2[0], true, false, 512, true);
    tk::dnn::Activation ch_relu_2_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d ch_2_conv2(&net, 126, 1, 1, 1, 1, 0, 0, classification_header2[1], false);
    tk::dnn::Layer *conf2[1] = {&ch_2_conv2};

    // //classification header 3
    tk::dnn::Route rout_ch_3(&net, header_3, 1);
    tk::dnn::Conv2d ch_3_conv1(&net, 256, 3, 3, 1, 1, 1, 1, classification_header3[0], true, false, 256, true);
    tk::dnn::Activation ch_relu_3_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d ch_3_conv2(&net, 126, 1, 1, 1, 1, 0, 0, classification_header3[1], false);
    tk::dnn::Layer *conf3[1] = {&ch_3_conv2};

    // //classification header 4
    tk::dnn::Route rout_ch_4(&net, header_4, 1);
    tk::dnn::Conv2d ch_4_conv1(&net, 256, 3, 3, 1, 1, 1, 1, classification_header4[0], true, false, 256, true);
    tk::dnn::Activation ch_relu_4_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d ch_4_conv2(&net, 126, 1, 1, 1, 1, 0, 0, classification_header4[1], false);
    tk::dnn::Layer *conf4[1] = {&ch_4_conv2};

    // //classification header 5
    tk::dnn::Route rout_ch_5(&net, header_5, 1);
    tk::dnn::Conv2d ch_5_conv(&net, 126, 1, 1, 1, 1, 0, 0, classification_header5, false);
    ch_5_conv.setFinal();
    tk::dnn::Layer *conf5[1] = {&ch_5_conv};

    //regression header 0
    tk::dnn::Route rout_rh_0(&net, header_0, 1);
    tk::dnn::Conv2d rh_0_conv1(&net, 576, 3, 3, 1, 1, 1, 1, regression_header0[0], true, false, 576, true);
    tk::dnn::Activation rh_relu_0_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d rh_0_conv2(&net, 24, 1, 1, 1, 1, 0, 0, regression_header0[1], false);
    tk::dnn::Layer *loc0[1] = {&rh_0_conv2};

    // //regression header 1
    tk::dnn::Route rout_rh_1(&net, header_1, 1);
    tk::dnn::Conv2d rh_1_conv1(&net, 1280, 3, 3, 1, 1, 1, 1, regression_header1[0], true, false, 1280, true);
    tk::dnn::Activation rh_relu_1_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d rh_1_conv2(&net, 24, 1, 1, 1, 1, 0, 0, regression_header1[1], false);
    tk::dnn::Layer *loc1[1] = {&rh_1_conv2};

    //regression header 2
    tk::dnn::Route rout_rh_2(&net, header_2, 1);
    tk::dnn::Conv2d rh_2_conv1(&net, 512, 3, 3, 1, 1, 1, 1, regression_header2[0], true, false, 512, true);
    tk::dnn::Activation rh_relu_2_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d rh_2_conv2(&net, 24, 1, 1, 1, 1, 0, 0, regression_header2[1], false);
    tk::dnn::Layer *loc2[1] = {&rh_2_conv2};

    //regression header 3
    tk::dnn::Route rout_rh_3(&net, header_3, 1);
    tk::dnn::Conv2d rh_3_conv1(&net, 256, 3, 3, 1, 1, 1, 1, regression_header3[0], true, false, 256, true);
    tk::dnn::Activation rh_relu_3_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d rh_3_conv2(&net, 24, 1, 1, 1, 1, 0, 0, regression_header3[1], false);
    tk::dnn::Layer *loc3[1] = {&rh_3_conv2};

    //regression header 4

    tk::dnn::Route rout_rh_4(&net, header_4, 1);
    tk::dnn::Conv2d rh_4_conv1(&net, 256, 3, 3, 1, 1, 1, 1, regression_header4[0], true, false, 256, true);
    tk::dnn::Activation rh_relu_4_1(&net, CUDNN_ACTIVATION_CLIPPED_RELU, 6);
    tk::dnn::Conv2d rh_4_conv2(&net, 24, 1, 1, 1, 1, 0, 0, regression_header4[1], false);
    tk::dnn::Layer *loc4[1] = {&rh_4_conv2};

    //regression header 5
    tk::dnn::Route rout_rh_5(&net, header_5, 1);
    tk::dnn::Conv2d rh_5_conv(&net, 24, 1, 1, 1, 1, 0, 0, regression_header5, false);
    rh_5_conv.setFinal();
    tk::dnn::Layer *loc5[1] = {&rh_5_conv};

    last = &rh_5_conv;

    //flatten all confidence
    tk::dnn::Route r_conf_0(&net, conf0, 1);
    tk::dnn::Flatten fl_c_0(&net);
    tk::dnn::Route r_conf_1(&net, conf1, 1);
    tk::dnn::Flatten fl_c_1(&net);
    tk::dnn::Route r_conf_2(&net, conf2, 1);
    tk::dnn::Flatten fl_c_2(&net);
    tk::dnn::Route r_conf_3(&net, conf3, 1);
    tk::dnn::Flatten fl_c_3(&net);
    tk::dnn::Route r_conf_4(&net, conf4, 1);
    tk::dnn::Flatten fl_c_4(&net);
    tk::dnn::Route r_conf_5(&net, conf5, 1);
    tk::dnn::Flatten fl_c_5(&net);

    // //flatten all locations
    tk::dnn::Route r_loc_0(&net, loc0, 1);
    tk::dnn::Flatten fl_l_0(&net);
    tk::dnn::Route r_loc_1(&net, loc1, 1);
    tk::dnn::Flatten fl_l_1(&net);
    tk::dnn::Route r_loc_2(&net, loc2, 1);
    tk::dnn::Flatten fl_l_2(&net);
    tk::dnn::Route r_loc_3(&net, loc3, 1);
    tk::dnn::Flatten fl_l_3(&net);
    tk::dnn::Route r_loc_4(&net, loc4, 1);
    tk::dnn::Flatten fl_l_4(&net);
    tk::dnn::Route r_loc_5(&net, loc5, 1);
    tk::dnn::Flatten fl_l_5(&net);

    // //concat confidence + softmax
    tk::dnn::Layer *confidences[6] = {&fl_c_0, &fl_c_1, &fl_c_2, &fl_c_3, &fl_c_4, &fl_c_5};
    tk::dnn::Route rout_conf(&net, confidences, 6);
    tk::dnn::dataDim_t olddim_c = net.layers[net.num_layers - 1]->output_dim;
    tk::dnn::dataDim_t dim_resh(1, olddim_c.c * olddim_c.h * olddim_c.w / classes, classes, 1, 1);

    tk::dnn::Reshape reshape_conf1(&net, dim_resh);
    tk::dnn::Flatten fl_l_6(&net);
    tk::dnn::dataDim_t newdim_c(1, classes, olddim_c.c * olddim_c.h * olddim_c.w / classes, 1, 1);

    tk::dnn::Reshape reshape_conf2(&net, newdim_c);

    tk::dnn::Softmax sm_1(&net, &newdim_c);
    sm_1.setFinal();
    // tk::dnn::Flatten fl_l_7(&net);
    // tk::dnn::Reshape reshape_conf3(&net,dim_resh, true);
    tk::dnn::Layer *conf = &sm_1;

    //concat locations
    tk::dnn::Layer *locations[6] = {&fl_l_0, &fl_l_1, &fl_l_2, &fl_l_3, &fl_l_4, &fl_l_5};
    tk::dnn::Route rout_loc(&net, locations, 6);
    tk::dnn::dataDim_t olddim_l = net.layers[net.num_layers - 1]->output_dim;
    tk::dnn::dataDim_t newdim_l(1, olddim_l.c * olddim_l.h * olddim_l.w / 4, 1, 4, 1);
    tk::dnn::Reshape reshape_loc(&net, newdim_l);
    reshape_loc.setFinal();
    tk::dnn::Layer *loc = &reshape_loc;

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    //printDeviceVector(64, data, true);

    //print network model
    net.print();

    // convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("mobilenetv2ssd"));

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TKDNN_TSTART
        net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }

    dnnType *cudnn_out1 = conf5[0]->dstData;
    tk::dnn::dataDim_t out_dim1 = conf5[0]->output_dim;
    dnnType *cudnn_out2 = loc5[0]->dstData;
    tk::dnn::dataDim_t out_dim2 = loc5[0]->output_dim;

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TKDNN_TSTART
        netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }

    dnnType *rt_out1 = (dnnType *)netRT.buffersRT[1];
    dnnType *rt_out2 = (dnnType *)netRT.buffersRT[2];
    dnnType *rt_out3 = (dnnType *)netRT.buffersRT[3];
    dnnType *rt_out4 = (dnnType *)netRT.buffersRT[4];

    printCenteredTitle(std::string(" RESNET CHECK RESULTS ").c_str(), '=', 30);
    dnnType *out1, *out1_h;
    int odim1 = out_dim1.tot();
    readBinaryFile(output_bin1, odim1, &out1_h, &out1);

    dnnType *out2, *out2_h;
    int odim2 = out_dim2.tot();
    readBinaryFile(output_bin2, odim2, &out2_h, &out2);
    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0; 

    std::cout << "CUDNN vs correct" << std::endl;
    ret_cudnn |= checkResult(odim1, cudnn_out1, out1) == 0 ? 0 : ERROR_CUDNN;
    ret_cudnn |= checkResult(odim2, cudnn_out2, out2) == 0 ? 0 : ERROR_CUDNN;

    std::cout << "TRT   vs correct" << std::endl;
    ret_tensorrt |= checkResult(odim1, rt_out1, out1) == 0 ? 0 : ERROR_TENSORRT;
    ret_tensorrt |= checkResult(odim2, rt_out2, out2) == 0 ? 0 : ERROR_TENSORRT;

    std::cout << "CUDNN vs TRT    " << std::endl;
    ret_cudnn_tensorrt |= checkResult(odim1, cudnn_out1, rt_out1) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    ret_cudnn_tensorrt |= checkResult(odim2, cudnn_out2, rt_out2) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Confidence CUDNN" << std::endl;
    printDeviceVector(64, conf->dstData, true);
    std::cout << "Locations CUDNN" << std::endl;
    printDeviceVector(64, loc->dstData, true);
    std::cout << "---------------------------------------------------" << std::endl;

    std::cout << "Confidence tensorRT" << std::endl;
    printDeviceVector(64, rt_out3, true);
    std::cout << "Locations tensorRT" << std::endl;
    printDeviceVector(64, rt_out4, true);
    std::cout << "---------------------------------------------------" << std::endl;

    std::cout << "CUDNN vs TRT    " << std::endl;
    ret_cudnn_tensorrt |= checkResult(conf->output_dim.tot(), conf->dstData, rt_out3) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    ret_cudnn_tensorrt |= checkResult(loc->output_dim.tot(), loc->dstData, rt_out4) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
