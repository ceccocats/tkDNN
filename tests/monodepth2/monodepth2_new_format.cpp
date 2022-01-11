#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tkdnn.h>

const char* encoder_conv1_bin = "monodepth2/layers/encoder/encoder-conv1.bin";
const char* encoder_bn1_bin = "monodepth2/layers/encoder/encoder-bn1.bin";

const char* encoder_layer1_conv_bin[] = {
        "monodepth2/layers/encoder/encoder-layer1-0-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer1-0-conv2.bin",
        "monodepth2/layers/encoder/encoder-layer1-1-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer1-1-conv2.bin",
};

const char* encoder_layer1_bn_bin[] = {
        "monodepth2/layers/encoder/encoder-layer1-0-bn1.bin",
        "monodepth2/layers/encoder/encoder-layer1-0-bn2.bin",
        "monodepth2/layers/encoder/encoder-layer1-1-bn1.bin",
        "monodepth2/layers/encoder/encoder-layer1-1-bn2.bin",
};



const char* encoder_layer2_conv_bin[] = {
        "monodepth2/layers/encoder/encoder-layer2-0-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer2-0-conv2.bin",
        "monodepth2/layers/encoder/encoder-layer2-0-downsample-0.bin",
        "monodepth2/layers/encoder/encoder-layer2-1-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer2-1-conv2.bin"
};

const char* encoder_layer2_bn_bin[] = {
        "monodepth2/layers/encoder/encoder-layer2-0-bn1.bin",
        "monodepth2/layers/encoder/encoder-layer2-0-bn2.bin",
        "monodepth2/layers/encoder/encoder-layer2-0-downsample-1.bin",
        "monodepth2/layers/encoder/encoder-layer2-1-bn1.bin",
        "monodepth2/layers/encoder/encoder-layer2-1-bn2.bin"
};

const char* encoder_layer3_conv_bin[]={
        "monodepth2/layers/encoder/encoder-layer3-0-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer3-0-conv2.bin",
        "monodepth2/layers/encoder/encoder-layer3-0-downsample-0.bin",
        "monodepth2/layers/encoder/encoder-layer3-1-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer3-1-conv2.bin"
};

const char* encoder_layer3_bn_bin[]={
        "monodepth2/layers/encoder/encoder-layer3-0-bn1.bin",
        "monodepth2/layers/encoder/encoder-layer3-0-bn2.bin",
        "monodepth2/layers/encoder/encoder-layer3-0-downsample-1.bin",
        "monodepth2/layers/encoder/encoder-layer3-1-bn1.bin",
        "monodepth2/layers/encoder/encoder-layer3-1-bn2.bin"
};

const char* encoder_layer4_conv_bin[] = {
        "monodepth2/layers/encoder/encoder-layer4-0-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer4-0-conv2.bin",
        "monodepth2/layers/encoder/encoder-layer4-0-downsample-0.bin",
        "monodepth2/layers/encoder/encoder-layer4-1-conv1.bin",
        "monodepth2/layers/encoder/encoder-layer4-1-conv2.bin"
};

const char* encoder_layer4_bn_bin[] = {
        "monodepth2/layers/encoder/encoder-layer4-0-bn1.bin",
        "monodepth2/layers/encoder/encoder-layer4-0-bn2.bin",
        "monodepth2/layers/encoder/encoder-layer4-0-downsample-1.bin",
        "monodepth2/layers/encoder/encoder-layer4-1-bn1.bin",
        "monodepth2/layers/encoder/encoder-layer4-1-bn2.bin"
};

const char* decoder_layer_bin[] = {
        "monodepth2/layers/depth_decoder/decoder-0-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-1-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-2-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-3-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-4-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-5-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-6-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-7-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-8-conv-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-9-conv-conv.bin"
};

const char* decoder_dispconv_layer_bin[] = {
        "monodepth2/layers/depth_decoder/decoder-10-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-11-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-12-conv.bin",
        "monodepth2/layers/depth_decoder/decoder-13-conv.bin"
};

const char* output_bin[] = {
        "monodepth2/debug/outputs/output-disp-0.bin",
        "monodepth2/debug/outputs/output-disp-1.bin",
        "monodepth2/debug/outputs/output-disp-2.bin",
        "monodepth2/debug/outputs/output-disp-3.bin"
};

const char* input_monodepth2_bin[] = {"monodepth2/debug/input.bin","monodepth2/debug/input2.bin"};

int main(){
    tk::dnn::dataDim_t dim(1,3,192,640,1);
    tk::dnn::Network net(dim);
    tk::dnn::Layer* encoder_conv = new tk::dnn::Conv2d(&net,64,7,7,2,2,3,3,encoder_conv1_bin);
    tk::dnn::Layer* encoder_bn = new tk::dnn::BatchNorm(&net,64,encoder_bn1_bin);
    tk::dnn::Layer* encoder_relu = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_maxpool = new tk::dnn::Pooling(&net,3,3,2,2,1,1,tk::dnn::POOLING_MAX);

    //layer-1
    tk::dnn::Layer* encoder_layer_1_0_conv_1 = new tk::dnn::Conv2d(&net,64,3,3,1,1,1,1,encoder_layer1_conv_bin[0]);
    tk::dnn::Layer* encoder_layer_1_0_bn_1 = new tk::dnn::BatchNorm(&net,64,encoder_layer1_bn_bin[0]);
    tk::dnn::Layer* encoder_relu_1 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_1_0_conv_2 = new tk::dnn::Conv2d(&net,64,3,3,1,1,1,1,encoder_layer1_conv_bin[1]);
    tk::dnn::Layer* encoder_layer_1_0_bn_2 = new tk::dnn::BatchNorm(&net,64,encoder_layer1_bn_bin[1]);
    tk::dnn::Layer* encoder_layer_1_0_shortcut_1 = new tk::dnn::Shortcut(&net,encoder_maxpool);
    tk::dnn::Layer* encoder_relu_2 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_1_1_conv_1 = new tk::dnn::Conv2d(&net,64,3,3,1,1,1,1,encoder_layer1_conv_bin[2]);
    tk::dnn::Layer* encoder_layer_1_1_bn_1 = new tk::dnn::BatchNorm(&net,64,encoder_layer1_bn_bin[2]);
    tk::dnn::Layer* encoder_relu_3 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_1_1_conv_2 = new tk::dnn::Conv2d(&net,64,3,3,1,1,1,1,encoder_layer1_conv_bin[3]);
    tk::dnn::Layer* encoder_layer_1_1_bn_2 = new tk::dnn::BatchNorm(&net,64,encoder_layer1_bn_bin[3]);
    tk::dnn::Layer* encoder_layer_1_1_shortcut_1 = new tk::dnn::Shortcut(&net,encoder_relu_2);
    tk::dnn::Layer* encoder_relu_4 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);

    //layer-2
    tk::dnn::Layer* encoder_layer_2_0_conv_1 = new tk::dnn::Conv2d(&net,128,3,3,2,2,1,1,encoder_layer2_conv_bin[0]);
    tk::dnn::Layer* encoder_layer_2_0_bn_1 = new tk::dnn::BatchNorm(&net,128,encoder_layer2_bn_bin[0]);
    tk::dnn::Layer* encoder_relu_5 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_2_0_conv_2 = new tk::dnn::Conv2d(&net,128,3,3,1,1,1,1,encoder_layer2_conv_bin[1]);
    tk::dnn::Layer* encoder_layer_2_0_bn_2 = new tk::dnn::BatchNorm(&net,128,encoder_layer2_bn_bin[1]);
    tk::dnn::Layer* encoder_layer_2_0_route = new tk::dnn::Route(&net,&encoder_relu_4,1);
    tk::dnn::Layer* encoder_layer_2_0_downsample_conv = new tk::dnn::Conv2d(&net,128,1,1,2,2,0,0,encoder_layer2_conv_bin[2]);
    tk::dnn::Layer* encoder_layer_2_0_downsample_bn = new tk::dnn::BatchNorm(&net,128,encoder_layer2_bn_bin[2]);
    tk::dnn::Layer* encoder_layer_2_0_shortcut = new tk::dnn::Shortcut(&net,encoder_layer_2_0_downsample_bn);
    tk::dnn::Layer* encoder_relu_6 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_2_1_conv_1 = new tk::dnn::Conv2d(&net,128,3,3,1,1,1,1,encoder_layer2_conv_bin[3]);
    tk::dnn::Layer* encoder_layer_2_1_bn_1 = new tk::dnn::BatchNorm(&net,128,encoder_layer2_bn_bin[3]);
    tk::dnn::Layer* encoder_relu_7 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_2_1_conv_2 = new tk::dnn::Conv2d(&net,128,3,3,1,1,1,1,encoder_layer2_conv_bin[4]);
    tk::dnn::Layer* encoder_layer_2_1_bn_2 = new tk::dnn::BatchNorm(&net,128,encoder_layer2_bn_bin[4]);
    tk::dnn::Layer* encoder_layer_2_1_shortcut = new tk::dnn::Shortcut(&net,encoder_relu_6);
    tk::dnn::Layer* encoder_relu_8 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);

    //layer-3
    tk::dnn::Layer* encoder_layer_3_0_conv_1 = new tk::dnn::Conv2d(&net,256,3,3,2,2,1,1,encoder_layer3_conv_bin[0]);
    tk::dnn::Layer* encoder_layer_3_0_bn_1 = new tk::dnn::BatchNorm(&net,256,encoder_layer3_bn_bin[0]);
    tk::dnn::Layer* encoder_relu_9 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_3_0_conv_2 = new tk::dnn::Conv2d(&net,256,3,3,1,1,1,1,encoder_layer3_conv_bin[1]);
    tk::dnn::Layer* encoder_layer_3_0_bn_2 = new tk::dnn::BatchNorm(&net,256,encoder_layer3_bn_bin[1]);
    tk::dnn::Layer* encoder_layer_3_0_route = new tk::dnn::Route(&net,&encoder_relu_8,1);
    tk::dnn::Layer* encoder_layer_3_0_downsample_conv = new tk::dnn::Conv2d(&net,256,1,1,2,2,0,0,encoder_layer3_conv_bin[2]);
    tk::dnn::Layer* encoder_layer_3_0_downsample_bn = new tk::dnn::BatchNorm(&net,256,encoder_layer3_bn_bin[2]);
    tk::dnn::Layer* encoder_layer_3_0_shortcut = new tk::dnn::Shortcut(&net,encoder_layer_3_0_bn_2);
    tk::dnn::Layer* encoder_relu_10 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_3_1_conv_1 = new tk::dnn::Conv2d(&net,256,3,3,1,1,1,1,encoder_layer3_conv_bin[3]);
    tk::dnn::Layer* encoder_layer_3_1_bn_1 = new tk::dnn::BatchNorm(&net,256,encoder_layer3_bn_bin[3]);
    tk::dnn::Layer* encoder_relu_11 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_3_1_conv_2 = new tk::dnn::Conv2d(&net,256,3,3,1,1,1,1,encoder_layer3_conv_bin[4]);
    tk::dnn::Layer* encoder_layer_3_1_bn_2 = new tk::dnn::BatchNorm(&net,256,encoder_layer3_bn_bin[4]);
    tk::dnn::Layer* encoder_layer_3_1_shortcut = new tk::dnn::Shortcut(&net,encoder_relu_10);
    tk::dnn::Layer* encoder_relu_12 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);

    //layer-4
    tk::dnn::Layer* encoder_layer_4_0_conv_1 = new tk::dnn::Conv2d(&net,512,3,3,2,2,1,1,encoder_layer4_conv_bin[0]);
    tk::dnn::Layer* encoder_layer_4_0_bn_1 = new tk::dnn::BatchNorm(&net,512,encoder_layer4_bn_bin[0]);
    tk::dnn::Layer* encoder_relu_13 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_4_0_conv_2 = new tk::dnn::Conv2d(&net,512,3,3,1,1,1,1,encoder_layer4_conv_bin[1]);
    tk::dnn::Layer* encoder_layer_4_0_bn_2 = new tk::dnn::BatchNorm(&net,512,encoder_layer4_bn_bin[1]);
    tk::dnn::Layer* encoder_layer_4_0_route = new tk::dnn::Route(&net,&encoder_relu_12,1);
    tk::dnn::Layer* encoder_layer_4_0_downsample_conv = new tk::dnn::Conv2d(&net,512,1,1,2,2,0,0,encoder_layer4_conv_bin[2]);
    tk::dnn::Layer* encoder_layer_4_0_downsample_bn = new tk::dnn::BatchNorm(&net,512,encoder_layer4_bn_bin[2]);
    tk::dnn::Layer* encoder_layer_4_0_shortcut = new tk::dnn::Shortcut(&net,encoder_layer_4_0_bn_2);
    tk::dnn::Layer* encoder_relu_14 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_4_1_conv_1 = new tk::dnn::Conv2d(&net,512,3,3,1,1,1,1,encoder_layer4_conv_bin[3]);
    tk::dnn::Layer* encoder_layer_4_1_bn_1 = new tk::dnn::BatchNorm(&net,512,encoder_layer4_bn_bin[3]);
    tk::dnn::Layer* encoder_relu_15 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);
    tk::dnn::Layer* encoder_layer_4_1_conv_2 = new tk::dnn::Conv2d(&net,512,3,3,1,1,1,1,encoder_layer4_conv_bin[4]);
    tk::dnn::Layer* encoder_layer_4_1_bn_2 = new tk::dnn::BatchNorm(&net,512,encoder_layer4_bn_bin[4]);
    tk::dnn::Layer* encoder_layer_4_1_shortcut = new tk::dnn::Shortcut(&net,encoder_relu_14);
    tk::dnn::Layer* encoder_relu_16 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_RELU);


    //decoder
    tk::dnn::Layer* decoder_reflection_padding_2d = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_4_0 = new tk::dnn::Conv2d(&net,256,3,3,1,1,0,0,decoder_layer_bin[0]);
    tk::dnn::Layer* decoder_elu = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_upsampling_2d = new tk::dnn::Upsample(&net,2);
    tk::dnn::Layer* concatenate_layer[2] = {decoder_upsampling_2d,encoder_relu_12};
    tk::dnn::Layer* decoder_concatenate = new tk::dnn::Route(&net,concatenate_layer,2);
    tk::dnn::Layer* decoder_reflection_padding_2d_1 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_4_1 = new tk::dnn::Conv2d(&net,256,3,3,1,1,0,0,decoder_layer_bin[1]);
    tk::dnn::Layer* decoder_elu_1 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_reflection_padding_2d_2 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_3_0 = new tk::dnn::Conv2d(&net,128,3,3,1,1,0,0,decoder_layer_bin[2]);
    tk::dnn::Layer* decoder_elu_2 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_upsampling_2d_1 = new tk::dnn::Upsample(&net,2);
    tk::dnn::Layer* concatenate_layer_1[2] = {decoder_upsampling_2d_1,encoder_relu_8};
    tk::dnn::Layer* decoder_concatenate_layer_1 = new tk::dnn::Route{&net,concatenate_layer_1,2};
    tk::dnn::Layer* decoder_reflection_padding_2d_3 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_3_1 = new tk::dnn::Conv2d(&net,128,3,3,1,1,0,0,decoder_layer_bin[3]);
    tk::dnn::Layer* decoder_elu_3 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_reflection_padding_2d_5 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_2_0 = new tk::dnn::Conv2d(&net,64,3,3,1,1,0,0,decoder_layer_bin[4]);
    tk::dnn::Layer* decoder_elu_4 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_upsampling_2d_2 = new tk::dnn::Upsample(&net,2);
    tk::dnn::Layer* concatenate_layer_2[2] = {decoder_upsampling_2d_2,encoder_relu_4};
    tk::dnn::Layer* decoder_concatenate_layer_2 = new tk::dnn::Route(&net,concatenate_layer_2,2);
    tk::dnn::Layer* decoder_reflection_padding_2d_6 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_2_1 = new tk::dnn::Conv2d(&net,64,3,3,1,1,0,0,decoder_layer_bin[5]);
    tk::dnn::Layer* decoder_elu_5 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_reflection_padding_2d_8 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_1_0 = new tk::dnn::Conv2d(&net,32,3,3,1,1,0,0,decoder_layer_bin[6]);
    tk::dnn::Layer* decoder_elu_6 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_upsampling_2d_3 = new tk::dnn::Upsample(&net,2);
    tk::dnn::Layer* concatenate_layer_3[2] = {decoder_upsampling_2d_3,encoder_relu};
    tk::dnn::Layer* decoder_concatenate_layer_3 = new tk::dnn::Route(&net,concatenate_layer_3,2);
    tk::dnn::Layer* decoder_reflection_padding_2d_9 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_1_1 = new tk::dnn::Conv2d(&net,32,3,3,1,1,0,0,decoder_layer_bin[7]);
    tk::dnn::Layer* decoder_elu_7 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_reflection_padding_2d_11 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_0_0 = new tk::dnn::Conv2d(&net,16,3,3,1,1,0,0,decoder_layer_bin[8]);
    tk::dnn::Layer* decoder_elu_8 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_upsampling_2d_4 = new tk::dnn::Upsample(&net,2);
    tk::dnn::Layer* decoder_reflection_padding_2d_12 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_upconv_0_1 = new tk::dnn::Conv2d(&net,16,3,3,1,1,0,0,decoder_layer_bin[9]);
    tk::dnn::Layer* decoder_elu_9 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_ELU);
    tk::dnn::Layer* decoder_reflection_padding_2d_13 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_dispconv_0 = new tk::dnn::Conv2d(&net,1,3,3,1,1,0,0,decoder_dispconv_layer_bin[0]);
    tk::dnn::Layer* disp0 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_SIGMOID);
    disp0->setFinal();

    tk::dnn::Layer* route_elu_7 = new tk::dnn::Route(&net,&decoder_elu_7,1);
    tk::dnn::Layer* decoder_reflection_padding_2d_10 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_dispconv_1 = new tk::dnn::Conv2d(&net,1,3,3,1,1,0,0,decoder_dispconv_layer_bin[1]);
    tk::dnn::Layer* disp1 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_SIGMOID);
    disp1->setFinal();

    tk::dnn::Layer* route_elu_5 = new tk::dnn::Route(&net,&decoder_elu_5,1);
    tk::dnn::Layer* decoder_reflection_padding_2d_7 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_dispconv_2 = new tk::dnn::Conv2d(&net,1,3,3,1,1,0,0,decoder_dispconv_layer_bin[2]);
    tk::dnn::Layer* disp2 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_SIGMOID);
    disp2->setFinal();

    tk::dnn::Layer* route_elu_3 = new tk::dnn::Route(&net,&decoder_elu_3,1);
    tk::dnn::Layer* decoder_reflection_padding_2d_4 = new tk::dnn::Padding(&net,1,1,tk::dnn::PADDING_MODE_REFLECTION);
    tk::dnn::Layer* decoder_dispconv_3 = new tk::dnn::Conv2d(&net,1,3,3,1,1,0,0,decoder_dispconv_layer_bin[3]);
    tk::dnn::Layer* disp3 = new tk::dnn::Activation(&net,CUDNN_ACTIVATION_SIGMOID);
    disp3->setFinal();

    dnnType *data;
    dnnType *input_H;
    readBinaryFile(input_monodepth2_bin[1],dim.tot(),&input_H,&data);
    std::cout<<"INPUT DIMENSIONS : "<<dim.tot()<<std::endl;

    net.print();

    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("monodepth2"));
    tk::dnn::dataDim_t dim1 = dim;
    dnnType *cudnn_out = nullptr;
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TKDNN_TSTART
        net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TKDNN_TSTART
        netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }
    tk::dnn::Layer *outs[4] = {disp0,disp1,disp2,disp3};
    std::cout<<std::endl<<std::endl;
    disp3->output_dim.print();
    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0;
    for(int i=0;i<4;i++){
        printCenteredTitle((std::string("MONODEPTH2 CHECK RESULTS ") + std::to_string(i) + " ").c_str(), '=', 30);
        outs[i]->output_dim.print();

        dnnType *out, *out_h;
        int odim = outs[i]->output_dim.tot();
        readBinaryFile(output_bin[i], odim, &out_h, &out);

        dnnType *cudnn_out, *rt_out;
        cudnn_out = outs[i]->dstData;
        rt_out = (dnnType *)netRT.buffersRT[i];
        std::cout<<"CUDNN vs correct";
        ret_cudnn |= checkResult(odim, cudnn_out, out) == 0 ? 0: ERROR_CUDNN;
        std::cout<<"TRT   vs correct";
        ret_tensorrt |= checkResult(odim, rt_out, out) == 0 ? 0 : ERROR_TENSORRT;
        std::cout<<"CUDNN vs TRT    ";
        ret_cudnn_tensorrt |= checkResult(odim, cudnn_out, rt_out) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    }


    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;

}