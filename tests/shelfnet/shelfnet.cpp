#include <iostream>
#include "tkdnn.h"


const char *output_bin1 = "shelfnet/debug/classification_headers-5.bin";
const char *output_bin2 = "shelfnet/debug/regression_headers-5.bin";
const char *input_bin = "shelfnet/debug/input.bin";

const char *backbone[] = {
    "shelfnet/layers/backbone-conv1.bin",
    "shelfnet/layers/backbone-layer1-0-conv1.bin",
    "shelfnet/layers/backbone-layer1-0-conv2.bin",
    "shelfnet/layers/backbone-layer1-1-conv1.bin",
    "shelfnet/layers/backbone-layer1-1-conv2.bin",
    "shelfnet/layers/backbone-layer2-0-conv1.bin",
    "shelfnet/layers/backbone-layer2-0-conv2.bin",
    "shelfnet/layers/backbone-layer2-0-downsample-0.bin",
    "shelfnet/layers/backbone-layer2-1-conv1.bin",
    "shelfnet/layers/backbone-layer2-1-conv2.bin",
    "shelfnet/layers/backbone-layer3-0-conv1.bin",
    "shelfnet/layers/backbone-layer3-0-conv2.bin",
    "shelfnet/layers/backbone-layer3-0-downsample-0.bin",
    "shelfnet/layers/backbone-layer3-1-conv1.bin",
    "shelfnet/layers/backbone-layer3-1-conv2.bin",
    "shelfnet/layers/backbone-layer4-0-conv1.bin",
    "shelfnet/layers/backbone-layer4-0-conv2.bin",
    "shelfnet/layers/backbone-layer4-0-downsample-0.bin",
    "shelfnet/layers/backbone-layer4-1-conv1.bin",
    "shelfnet/layers/backbone-layer4-1-conv2.bin"};

const char *conv_out[] = {
    "shelfnet/layers/conv_out16-conv-conv.bin",
    "shelfnet/layers/conv_out16-conv_out.bin",
    "shelfnet/layers/conv_out32-conv-conv.bin",
    "shelfnet/layers/conv_out32-conv_out.bin",
    "shelfnet/layers/conv_out-conv-conv.bin",
    "shelfnet/layers/conv_out-conv_out.bin"};

const char *decoder[] = {
    "shelfnet/layers/decoder-bottom-conv1.bin",
    "shelfnet/layers/decoder-up_conv_list-0-conv_atten.bin",
    "shelfnet/layers/decoder-up_conv_list-0-conv-conv.bin",
    "shelfnet/layers/decoder-up_conv_list-1-conv_atten.bin",
    "shelfnet/layers/decoder-up_conv_list-1-conv-conv.bin",
    "shelfnet/layers/decoder-up_dense_list-0-conv.bin",
    "shelfnet/layers/decoder-up_dense_list-1-conv.bin"};

    
const char *ladder[] = {
    "shelfnet/layers/ladder-bottom-conv1.bin",
    "shelfnet/layers/ladder-down_conv_list-0.bin",
    "shelfnet/layers/ladder-down_conv_list-1.bin",
    "shelfnet/layers/ladder-down_module_list-0-conv1.bin",
    "shelfnet/layers/ladder-down_module_list-1-conv1.bin",
    "shelfnet/layers/ladder-inconv-conv1.bin",
    "shelfnet/layers/ladder-up_conv_list-0-conv_atten.bin",
    "shelfnet/layers/ladder-up_conv_list-0-conv-conv.bin",
    "shelfnet/layers/ladder-up_conv_list-1-conv_atten.bin",
    "shelfnet/layers/ladder-up_conv_list-1-conv-conv.bin",
    "shelfnet/layers/ladder-up_dense_list-0-conv.bin",
    "shelfnet/layers/ladder-up_dense_list-1-conv.bin"};

const char *trans[] = {
    "shelfnet/layers/trans1-conv.bin",
    "shelfnet/layers/trans2-conv.bin",
    "shelfnet/layers/trans3-conv.bin"};
int main()
{

    // downloadWeightsifDoNotExist(input_bin, "shelfnet", "https://cloud.hipert.unimore.it/s/x4ZfxBKN23zAJQp/download");

    int classes = 19;

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 1024, 1024, 1);
    tk::dnn::Network net(dim);

    int bi = 0;
    new tk::dnn::Conv2d(&net, 64, 7, 7, 2, 2, 3, 3, backbone[bi++], true);
    new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer* last = new tk::dnn::Pooling (&net, 3, 3, 2, 2, 1, 1, tk::dnn::POOLING_MAX);


    
    for(int i=0; i<2; ++i){
        new tk::dnn::Conv2d (&net, 64, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY);
        new tk::dnn::Conv2d (&net, 64, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        new tk::dnn::Shortcut(&net, last);
        last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);
    }

    std::vector<tk::dnn::Layer*> features;
    for(int i=0;i<3;++i){
        int out_channel = pow(2,7+i);
        std::cout<<out_channel<<std::endl;
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 2, 2, 1, 1, backbone[bi++], true);
        new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY);
        tk::dnn::Layer* bn2 = new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        new tk::dnn::Route(&net, &last, 1);
        new tk::dnn::Conv2d (&net, out_channel, 1, 1, 2, 2, 0, 0, backbone[bi++], true);
        new tk::dnn::Shortcut(&net, bn2);
        last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);

        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY);
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, backbone[bi++], true);

        if(i != 2)
        
        {new tk::dnn::Shortcut(&net, last);
        last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);
        features.push_back(last);}
    }

    // for(int i=0; i<features.size(); ++i){
    //     new tk::dnn::Route(&net, &features[i], 1);
    //     int out_channel = pow(2,6+i);
    //     new tk::dnn::Conv2d (&net, out_channel, 1, 1, 1, 1, 0, 0, trans[i], true);
    //     new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY);
    // }

    

    const char *output_bin = "shelfnet/debug/backbone-layer4-1-bn2.bin";

    

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    std::cout<<"Input:"<<std::endl;
    // printDeviceVector(64, data, true);

    //print network model
    net.print();

    // // convert network to tensorRT
    // tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("shelfnet"));

    tk::dnn::dataDim_t dim1 = dim; //input dim
    dnnType *cudnn_out = nullptr;
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TKDNN_TSTART
        cudnn_out = net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }

    // tk::dnn::dataDim_t out_dim1 = conf5[0]->output_dim;
    // dnnType *cudnn_out2 = loc5[0]->dstData;
    // tk::dnn::dataDim_t out_dim2 = loc5[0]->output_dim;

    // tk::dnn::dataDim_t dim2 = dim;
    // printCenteredTitle(" TENSORRT inference ", '=', 30);
    // {
    //     dim2.print();
    //     TKDNN_TSTART
    //     netRT.infer(dim2, data);
    //     TKDNN_TSTOP
    //     dim2.print();
    // }

    // dnnType *rt_out1 = (dnnType *)netRT.buffersRT[1];
    // dnnType *rt_out2 = (dnnType *)netRT.buffersRT[2];
    // dnnType *rt_out3 = (dnnType *)netRT.buffersRT[3];
    // dnnType *rt_out4 = (dnnType *)netRT.buffersRT[4];

    printCenteredTitle(std::string(" RESNET CHECK RESULTS ").c_str(), '=', 30);
    dnnType *out1, *out1_h;
    int odim1 = dim1.tot();
    readBinaryFile(output_bin, odim1, &out1_h, &out1);

    printDeviceVector(64, out1);

    // dnnType *out2, *out2_h;
    // int odim2 = out_dim2.tot();
    // readBinaryFile(output_bin2, odim2, &out2_h, &out2);
    // int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0; 

    std::cout << "CUDNN vs correct" << std::endl;
    checkResult(odim1, cudnn_out, out1, true, 20) == 0 ? 0 : ERROR_CUDNN;

    // std::cout << "TRT   vs correct" << std::endl;
    // checkResult(odim1, rt_out1, out1) == 0 ? 0 : ERROR_TENSORRT;
    // ret_tensorrt |= checkResult(odim2, rt_out2, out2) == 0 ? 0 : ERROR_TENSORRT;

    // std::cout << "CUDNN vs TRT    " << std::endl;
    // ret_cudnn_tensorrt |= checkResult(odim1, cudnn_out1, rt_out1) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    // ret_cudnn_tensorrt |= checkResult(odim2, cudnn_out2, rt_out2) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    // std::cout << "---------------------------------------------------" << std::endl;
    // std::cout << "Confidence CUDNN" << std::endl;
    // printDeviceVector(64, conf->dstData, true);
    // std::cout << "Locations CUDNN" << std::endl;
    // printDeviceVector(64, loc->dstData, true);
    // std::cout << "---------------------------------------------------" << std::endl;

    // std::cout << "Confidence tensorRT" << std::endl;
    // printDeviceVector(64, rt_out3, true);
    // std::cout << "Locations tensorRT" << std::endl;
    // printDeviceVector(64, rt_out4, true);
    // std::cout << "---------------------------------------------------" << std::endl;

    // std::cout << "CUDNN vs TRT    " << std::endl;
    // ret_cudnn_tensorrt |= checkResult(conf->output_dim.tot(), conf->dstData, rt_out3) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    // ret_cudnn_tensorrt |= checkResult(loc->output_dim.tot(), loc->dstData, rt_out4) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    // return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
