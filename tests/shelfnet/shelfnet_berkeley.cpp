#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"
#include "NetworkViz.h"


const char *input_bin = "shelfnet_berkeley/debug/input.bin";

const char *backbone[] = {
    "shelfnet_berkeley/layers/backbone-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer1-0-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer1-0-conv2.bin",
    "shelfnet_berkeley/layers/backbone-layer1-1-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer1-1-conv2.bin",
    "shelfnet_berkeley/layers/backbone-layer2-0-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer2-0-conv2.bin",
    "shelfnet_berkeley/layers/backbone-layer2-0-downsample-0.bin",
    "shelfnet_berkeley/layers/backbone-layer2-1-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer2-1-conv2.bin",
    "shelfnet_berkeley/layers/backbone-layer3-0-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer3-0-conv2.bin",
    "shelfnet_berkeley/layers/backbone-layer3-0-downsample-0.bin",
    "shelfnet_berkeley/layers/backbone-layer3-1-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer3-1-conv2.bin",
    "shelfnet_berkeley/layers/backbone-layer4-0-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer4-0-conv2.bin",
    "shelfnet_berkeley/layers/backbone-layer4-0-downsample-0.bin",
    "shelfnet_berkeley/layers/backbone-layer4-1-conv1.bin",
    "shelfnet_berkeley/layers/backbone-layer4-1-conv2.bin"};

const char *conv_out[] = {
    "shelfnet_berkeley/layers/conv_out-conv-conv.bin",
    "shelfnet_berkeley/layers/conv_out-conv_out.bin",
    "shelfnet_berkeley/layers/conv_out16-conv-conv.bin",
    "shelfnet_berkeley/layers/conv_out16-conv_out.bin",
    "shelfnet_berkeley/layers/conv_out32-conv-conv.bin",
    "shelfnet_berkeley/layers/conv_out32-conv_out.bin"
    };

const char *decoder[] = {
    "shelfnet_berkeley/layers/decoder-bottom-conv1.bin",
    "shelfnet_berkeley/layers/decoder-bottom-conv12.bin",
    "shelfnet_berkeley/layers/decoder-up_conv_list-0-conv-conv.bin",
    "shelfnet_berkeley/layers/decoder-up_conv_list-0-conv_atten.bin",
    "shelfnet_berkeley/layers/decoder-up_dense_list-0-conv.bin",
    "shelfnet_berkeley/layers/decoder-up_conv_list-1-conv-conv.bin",
    "shelfnet_berkeley/layers/decoder-up_conv_list-1-conv_atten.bin",
    "shelfnet_berkeley/layers/decoder-up_dense_list-1-conv.bin"
    };

    
const char *ladder[] = {
    "shelfnet_berkeley/layers/ladder-inconv-conv1.bin",
    "shelfnet_berkeley/layers/ladder-inconv-conv12.bin",
    "shelfnet_berkeley/layers/ladder-down_module_list-0-conv1.bin",
    "shelfnet_berkeley/layers/ladder-down_module_list-0-conv12.bin",
    "shelfnet_berkeley/layers/ladder-down_conv_list-0.bin",

    "shelfnet_berkeley/layers/ladder-down_module_list-1-conv1.bin",
    "shelfnet_berkeley/layers/ladder-down_module_list-1-conv12.bin",
    "shelfnet_berkeley/layers/ladder-down_conv_list-1.bin",

    "shelfnet_berkeley/layers/ladder-bottom-conv1.bin",
    "shelfnet_berkeley/layers/ladder-bottom-conv12.bin",
    
    
    
    "shelfnet_berkeley/layers/ladder-up_conv_list-0-conv-conv.bin",
    "shelfnet_berkeley/layers/ladder-up_conv_list-0-conv_atten.bin",
    "shelfnet_berkeley/layers/ladder-up_dense_list-0-conv.bin",

    
    "shelfnet_berkeley/layers/ladder-up_conv_list-1-conv-conv.bin",
    "shelfnet_berkeley/layers/ladder-up_conv_list-1-conv_atten.bin",
    "shelfnet_berkeley/layers/ladder-up_dense_list-1-conv.bin"};

const char *trans[] = {
    "shelfnet_berkeley/layers/trans1-conv.bin",
    "shelfnet_berkeley/layers/trans2-conv.bin",
    "shelfnet_berkeley/layers/trans3-conv.bin"};
int main()
{

    downloadWeightsifDoNotExist(input_bin, "shelfnet_berkeley", "https://cloud.hipert.unimore.it/s/m92e7QdD9gYMF7f/download");

    int classes = 20;

    // Network layout    
    tk::dnn::dataDim_t dim(1, 3, 736, 1280, 1);
    tk::dnn::Network net(dim);

    int bi = 0, di = 0, li = 0, ci = 0;
    new tk::dnn::Conv2d(&net, 64, 7, 7, 2, 2, 3, 3, backbone[bi++], true);
    new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
    tk::dnn::Layer* last = new tk::dnn::Pooling (&net, 3, 3, 2, 2, 1, 1, tk::dnn::POOLING_MAX);


    
    for(int i=0; i<2; ++i){
        new tk::dnn::Conv2d (&net, 64, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        new tk::dnn::Conv2d (&net, 64, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        new tk::dnn::Shortcut(&net, last);
        last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);
    }

    std::vector<tk::dnn::Layer*> features;
    for(int i=0;i<3;++i){
        int out_channel = pow(2,7+i);
        std::cout<<out_channel<<std::endl;
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 2, 2, 1, 1, backbone[bi++], true);
        new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        tk::dnn::Layer* bn2 = new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        new tk::dnn::Route(&net, &last, 1);
        new tk::dnn::Conv2d (&net, out_channel, 1, 1, 2, 2, 0, 0, backbone[bi++], true);
        new tk::dnn::Shortcut(&net, bn2);
        last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);

        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, backbone[bi++], true);
        
        new tk::dnn::Shortcut(&net, last);
        last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);
        features.push_back(last);
    }

    for(int i=0; i<features.size(); ++i){
        new tk::dnn::Route(&net, &features[i], 1);
        int out_channel = pow(2,6+i);
        new tk::dnn::Conv2d (&net, out_channel, 1, 1, 1, 1, 0, 0, trans[i], true);
        features[i] = new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
    }

    //DECODER
    
    last = features[2];
    std::vector<tk::dnn::Layer*> up_out;
    //bottom
    new tk::dnn::Conv2d (&net, 256, 3, 3, 1, 1, 1, 1, decoder[di++], true, false, 1, true);
    new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
    new tk::dnn::Conv2d (&net, 256, 3, 3, 1, 1, 1, 1, decoder[di++], true, false, 1, true);
    new tk::dnn::Shortcut(&net, last);
    last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);
    up_out.push_back(last);

    for(int i=0; i<2; ++i){
        int out_channel = pow(2,7-i);
        //up-conv
        std::cout<<out_channel<<std::endl;
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, decoder[di++], true);
        last = new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        
        new tk::dnn::Pooling(&net, last->output_dim.w, last->output_dim.h, last->output_dim.w, last->output_dim.h, 0, 0, tk::dnn::POOLING_AVERAGE);
        new tk::dnn::Conv2d (&net, out_channel, 1, 1, 1, 1, 0, 0, decoder[di++], true);
        
        tk::dnn::Layer* act = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_SIGMOID);
        new tk::dnn::Route(&net, &last, 1);
        new tk::dnn::Shortcut(&net, act, true);

        //interpolate
        new tk::dnn::Resize(&net, 1,2,2);
        new tk::dnn::Shortcut(&net, features[1-i]);

        //up-dense
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, decoder[di++], true);
        last = new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        up_out.push_back(last);
    }

    //LADDER

    std::vector<tk::dnn::Layer*> down_out;
    new tk::dnn::Conv2d (&net, 64, 3, 3, 1, 1, 1, 1, ladder[li++], true, false, 1, true);
    new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
    new tk::dnn::Conv2d (&net, 64, 3, 3, 1, 1, 1, 1, ladder[li++], true, false, 1, true);
    new tk::dnn::Shortcut(&net, last);
    new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);
    
    for(int i=0; i<2;++i){
        int out_channel = pow(2,6+i);
        tk::dnn::Layer* l_last = new tk::dnn::Shortcut(&net, up_out[2-i]);
    
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, ladder[li++], true, false, 1, true);
        new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, ladder[li++], true, false, 1, true);
        new tk::dnn::Shortcut(&net, l_last);
        l_last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);
        down_out.push_back(l_last);

        new tk::dnn::Conv2d (&net, out_channel*2, 3, 3, 2, 2, 1, 1, ladder[li++], false);
        last = new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.0f); //should be ReLU
    }

    new tk::dnn::Conv2d (&net, 256, 3, 3, 1, 1, 1, 1, ladder[li++], true, false, 1, true);
    new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
    new tk::dnn::Conv2d (&net, 256, 3, 3, 1, 1, 1, 1, ladder[li++], true, false, 1, true);
    new tk::dnn::Shortcut(&net, last);
    last = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_RELU);
    up_out.clear();
    up_out.push_back(last);

    for(int i=0; i<2; ++i){
        int out_channel = pow(2,7-i);
        //up-conv
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, ladder[li++], true);
        last = new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        
        new tk::dnn::Pooling(&net, last->output_dim.w, last->output_dim.h, last->output_dim.w, last->output_dim.h, 0, 0, tk::dnn::POOLING_AVERAGE);
        new tk::dnn::Conv2d (&net, out_channel, 1, 1, 1, 1, 0, 0, ladder[li++], true);
        
        tk::dnn::Layer* act = new tk::dnn::Activation (&net, CUDNN_ACTIVATION_SIGMOID);
        new tk::dnn::Route(&net, &last, 1);
        new tk::dnn::Shortcut(&net, act, true);

        //interpolate
        new tk::dnn::Resize(&net, 1,2,2);
        new tk::dnn::Shortcut(&net, down_out[1-i]);

        // //up-dense
        new tk::dnn::Conv2d (&net, out_channel, 3, 3, 1, 1, 1, 1, ladder[li++], true);
        last = new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        up_out.push_back(last);
    }


    // for(int i=2;i>=0;--i){
        // new tk::dnn::Route(&net, &up_out[i], 1);
        new tk::dnn::Conv2d (&net, 64, 3, 3, 1, 1, 1, 1, conv_out[ci++], true);
        new tk::dnn::Activation (&net, tk::dnn::ACTIVATION_LEAKY, 0.0f, 0.01);
        new tk::dnn::Conv2d (&net, classes, 3, 3, 1, 1, 1, 1, conv_out[ci++], false);
        /*up_out[i] =*/ new tk::dnn::Resize(&net, classes, net.input_dim.h, net.input_dim.w, true, tk::dnn::ResizeMode_t::LINEAR);
    // }

    new tk::dnn::Softmax(&net);
    
    const char *output_bin = "shelfnet_berkeley/debug/softmax.bin";
    
    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    std::cout<<"Input:"<<std::endl;

    //print network model
    net.print();

    // // convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("shelfnet_berkeley"));

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

    printCenteredTitle(std::string(" CHECK RESULTS ").c_str(), '=', 30);
    dnnType *out1, *out1_h;
    int odim1 = dim1.tot();
    readBinaryFile(output_bin, odim1, &out1_h, &out1);

    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0; 
    std::cout << "CUDNN vs correct" << std::endl;
    ret_cudnn |= checkResult(odim1, cudnn_out, out1, true, 20) == 0 ? 0 : ERROR_CUDNN;

    std::cout << "TRT   vs correct" << std::endl;
    ret_tensorrt |=checkResult(odim1, rt_out1, out1) == 0 ? 0 : ERROR_TENSORRT;

    std::cout << "CUDNN vs TRT    " << std::endl;
    ret_cudnn_tensorrt |= checkResult(odim1, cudnn_out, rt_out1) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    
    cv::Mat viz = vizLayer2Mat(&net, net.num_layers-1);
    cv::imwrite("test.png", viz);

    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
