#include <iostream>
#include "tkdnn.h"

const char *input_bin = "resnet101/debug/input.bin";
const char *conv1_bin = "resnet101/layers/conv1.bin";

//layer1
const char *layer1_bin[]={
"resnet101/layers/layer1-0-conv1.bin",
"resnet101/layers/layer1-0-conv2.bin",
"resnet101/layers/layer1-0-conv3.bin",
"resnet101/layers/layer1-0-downsample-0.bin",

"resnet101/layers/layer1-1-conv1.bin",
"resnet101/layers/layer1-1-conv2.bin",
"resnet101/layers/layer1-1-conv3.bin",

"resnet101/layers/layer1-2-conv1.bin",
"resnet101/layers/layer1-2-conv2.bin",
"resnet101/layers/layer1-2-conv3.bin"};
                        

//layer2
const char *layer2_bin[]={
"resnet101/layers/layer2-0-conv1.bin",
"resnet101/layers/layer2-0-conv2.bin",
"resnet101/layers/layer2-0-conv3.bin",
"resnet101/layers/layer2-0-downsample-0.bin",

"resnet101/layers/layer2-1-conv1.bin",
"resnet101/layers/layer2-1-conv2.bin",
"resnet101/layers/layer2-1-conv3.bin",

"resnet101/layers/layer2-2-conv1.bin",
"resnet101/layers/layer2-2-conv2.bin",
"resnet101/layers/layer2-2-conv3.bin",

"resnet101/layers/layer2-3-conv1.bin",
"resnet101/layers/layer2-3-conv2.bin",
"resnet101/layers/layer2-3-conv3.bin"
};
//layer3
const char *layer3_bin[]={
"resnet101/layers/layer3-0-conv1.bin",
"resnet101/layers/layer3-0-conv2.bin",
"resnet101/layers/layer3-0-conv3.bin",
"resnet101/layers/layer3-0-downsample-0.bin",

"resnet101/layers/layer3-1-conv1.bin",
"resnet101/layers/layer3-1-conv2.bin",
"resnet101/layers/layer3-1-conv3.bin",

"resnet101/layers/layer3-2-conv1.bin",
"resnet101/layers/layer3-2-conv2.bin",
"resnet101/layers/layer3-2-conv3.bin",

"resnet101/layers/layer3-3-conv1.bin",
"resnet101/layers/layer3-3-conv2.bin",
"resnet101/layers/layer3-3-conv3.bin",

"resnet101/layers/layer3-4-conv1.bin",
"resnet101/layers/layer3-4-conv2.bin",
"resnet101/layers/layer3-4-conv3.bin",

"resnet101/layers/layer3-5-conv1.bin",
"resnet101/layers/layer3-5-conv2.bin",
"resnet101/layers/layer3-5-conv3.bin",

"resnet101/layers/layer3-6-conv1.bin",
"resnet101/layers/layer3-6-conv2.bin",
"resnet101/layers/layer3-6-conv3.bin",

"resnet101/layers/layer3-7-conv1.bin",
"resnet101/layers/layer3-7-conv2.bin",
"resnet101/layers/layer3-7-conv3.bin",

"resnet101/layers/layer3-8-conv1.bin",
"resnet101/layers/layer3-8-conv2.bin",
"resnet101/layers/layer3-8-conv3.bin",

"resnet101/layers/layer3-9-conv1.bin",
"resnet101/layers/layer3-9-conv2.bin",
"resnet101/layers/layer3-9-conv3.bin",

"resnet101/layers/layer3-10-conv1.bin",
"resnet101/layers/layer3-10-conv2.bin",
"resnet101/layers/layer3-10-conv3.bin",

"resnet101/layers/layer3-11-conv1.bin",
"resnet101/layers/layer3-11-conv2.bin",
"resnet101/layers/layer3-11-conv3.bin",

"resnet101/layers/layer3-12-conv1.bin",
"resnet101/layers/layer3-12-conv2.bin",
"resnet101/layers/layer3-12-conv3.bin",

"resnet101/layers/layer3-13-conv1.bin",
"resnet101/layers/layer3-13-conv2.bin",
"resnet101/layers/layer3-13-conv3.bin",

"resnet101/layers/layer3-14-conv1.bin",
"resnet101/layers/layer3-14-conv2.bin",
"resnet101/layers/layer3-14-conv3.bin",

"resnet101/layers/layer3-15-conv1.bin",
"resnet101/layers/layer3-15-conv2.bin",
"resnet101/layers/layer3-15-conv3.bin",

"resnet101/layers/layer3-16-conv1.bin",
"resnet101/layers/layer3-16-conv2.bin",
"resnet101/layers/layer3-16-conv3.bin",

"resnet101/layers/layer3-17-conv1.bin",
"resnet101/layers/layer3-17-conv2.bin",
"resnet101/layers/layer3-17-conv3.bin",

"resnet101/layers/layer3-18-conv1.bin",
"resnet101/layers/layer3-18-conv2.bin",
"resnet101/layers/layer3-18-conv3.bin",

"resnet101/layers/layer3-19-conv1.bin",
"resnet101/layers/layer3-19-conv2.bin",
"resnet101/layers/layer3-19-conv3.bin",

"resnet101/layers/layer3-20-conv1.bin",
"resnet101/layers/layer3-20-conv2.bin",
"resnet101/layers/layer3-20-conv3.bin",

"resnet101/layers/layer3-21-conv1.bin",
"resnet101/layers/layer3-21-conv2.bin",
"resnet101/layers/layer3-21-conv3.bin",

"resnet101/layers/layer3-22-conv1.bin",
"resnet101/layers/layer3-22-conv2.bin",
"resnet101/layers/layer3-22-conv3.bin"};


//layer4
const char *layer4_bin[]={
"resnet101/layers/layer4-0-conv1.bin",
"resnet101/layers/layer4-0-conv2.bin",
"resnet101/layers/layer4-0-conv3.bin",
"resnet101/layers/layer4-0-downsample-0.bin",

"resnet101/layers/layer4-1-conv1.bin",
"resnet101/layers/layer4-1-conv2.bin",
"resnet101/layers/layer4-1-conv3.bin",

"resnet101/layers/layer4-2-conv1.bin",
"resnet101/layers/layer4-2-conv2.bin",
"resnet101/layers/layer4-2-conv3.bin"};

//final
const char *fc_bin = "resnet101/layers/fc.bin";

const char *output_bin = "resnet101/debug/fc.bin";

int main()
{

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 224, 224, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d conv1(&net, 64, 7, 7, 2, 2, 3, 3, conv1_bin, true);
    tk::dnn::Activation relu3(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Pooling maxpool4(&net, 3, 3, 2, 2, 1, 1, tk::dnn::POOLING_MAX);

    //layer 1
    int id_layer1_bin = 0;
    tk::dnn::Layer *last = &maxpool4;
    for(int i=0; i<3;i++)
    {
        tk::dnn::Conv2d     *layer1_0_conv1 = new tk::dnn::Conv2d(&net, 64, 1, 1, 1, 1, 0, 0, layer1_bin[id_layer1_bin++], true);
        tk::dnn::Activation *relu1_0_1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        tk::dnn::Conv2d     *layer1_0_conv2 = new tk::dnn::Conv2d(&net, 64, 3, 3, 1, 1, 1, 1, layer1_bin[id_layer1_bin++], true);
        tk::dnn::Activation *relu1_0_2      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        tk::dnn::Conv2d     *layer1_0_conv3 = new tk::dnn::Conv2d(&net, 256, 1, 1, 1, 1, 0, 0, layer1_bin[id_layer1_bin++], true);
        if(i==0) {
            tk::dnn::Layer    *route_1_0_layers[1] = { last };
            tk::dnn::Route    *route_1_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
            tk::dnn::Conv2d   *layer1_0_downsample_0 = new tk::dnn::Conv2d(&net, 256, 1, 1, 1, 1, 0, 0, layer1_bin[id_layer1_bin++], true);
            tk::dnn::Shortcut *s1_0                  = new tk::dnn::Shortcut(&net, layer1_0_conv3);
        } else {
            tk::dnn::Shortcut *s1_0                  = new tk::dnn::Shortcut(&net, last);
        }
        tk::dnn::Activation   *layer1_0_relu = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        last = layer1_0_relu;
    }

    // tk::dnn::Activation *last_activation = (tk::dnn::Activation *) net.layers[net.num_layers-1];
    // layer 2
    int id_layer2_bin = 0;
    for(int i=0; i<4;i++)
    {
        tk::dnn::Conv2d     *layer1_0_conv1 = new tk::dnn::Conv2d(&net, 128, 1, 1, 1, 1, 0, 0, layer2_bin[id_layer2_bin++], true);
        tk::dnn::Activation *relu1_0_1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        tk::dnn::Conv2d     *layer1_0_conv2;
        if(i==0)
            layer1_0_conv2 = new tk::dnn::Conv2d(&net, 128, 3, 3, 2, 2, 1, 1, layer2_bin[id_layer2_bin++], true);
        else
            layer1_0_conv2 = new tk::dnn::Conv2d(&net, 128, 3, 3, 1, 1, 1, 1, layer2_bin[id_layer2_bin++], true);
        
        tk::dnn::Activation *relu1_0_2      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        tk::dnn::Conv2d     *layer1_0_conv3 = new tk::dnn::Conv2d(&net, 512, 1, 1, 1, 1, 0, 0, layer2_bin[id_layer2_bin++], true);
        if(i==0) 
        {
            tk::dnn::Layer    *route_1_0_layers[1] = { last };
            tk::dnn::Route    *route_1_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
            tk::dnn::Conv2d   *layer1_0_downsample_0 = new tk::dnn::Conv2d(&net, 512, 1, 1, 2, 2, 0, 0, layer2_bin[id_layer2_bin++], true);
            tk::dnn::Shortcut *s1_0                  = new tk::dnn::Shortcut(&net, layer1_0_conv3);
        } 
        else 
        {
            tk::dnn::Shortcut *s1_0                  = new tk::dnn::Shortcut(&net, last);
        }
        tk::dnn::Activation   *layer1_0_relu = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        last = layer1_0_relu;
    }
    
    // layer 3
    int id_layer3_bin = 0;
    for(int i=0; i<23;i++)
    {
        tk::dnn::Conv2d     *layer1_0_conv1 = new tk::dnn::Conv2d(&net, 256, 1, 1, 1, 1, 0, 0, layer3_bin[id_layer3_bin++], true);
        tk::dnn::Activation *relu1_0_1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        tk::dnn::Conv2d     *layer1_0_conv2;
        if(i==0)
            layer1_0_conv2 = new tk::dnn::Conv2d(&net, 256, 3, 3, 2, 2, 1, 1, layer3_bin[id_layer3_bin++], true);
        else
            layer1_0_conv2 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, layer3_bin[id_layer3_bin++], true);
        
        tk::dnn::Activation *relu1_0_2      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        tk::dnn::Conv2d     *layer1_0_conv3 = new tk::dnn::Conv2d(&net, 1024, 1, 1, 1, 1, 0, 0, layer3_bin[id_layer3_bin++], true);
        if(i==0) 
        {
            tk::dnn::Layer    *route_1_0_layers[1] = { last };
            tk::dnn::Route    *route_1_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
            tk::dnn::Conv2d   *layer1_0_downsample_0 = new tk::dnn::Conv2d(&net, 1024, 1, 1, 2, 2, 0, 0, layer3_bin[id_layer3_bin++], true);
            tk::dnn::Shortcut *s1_0                  = new tk::dnn::Shortcut(&net, layer1_0_conv3);
        } 
        else 
        {
            tk::dnn::Shortcut *s1_0                  = new tk::dnn::Shortcut(&net, last);
        }
        tk::dnn::Activation   *layer1_0_relu = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        last = layer1_0_relu;
    }

    // layer 4
    int id_layer4_bin = 0;
    for(int i=0; i<3;i++)
    {
        tk::dnn::Conv2d     *layer1_0_conv1 = new tk::dnn::Conv2d(&net, 512, 1, 1, 1, 1, 0, 0, layer4_bin[id_layer4_bin++], true);
        tk::dnn::Activation *relu1_0_1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        tk::dnn::Conv2d     *layer1_0_conv2;
        if(i==0)
            layer1_0_conv2 = new tk::dnn::Conv2d(&net, 512, 3, 3, 2, 2, 1, 1, layer4_bin[id_layer4_bin++], true);
        else
            layer1_0_conv2 = new tk::dnn::Conv2d(&net, 512, 3, 3, 1, 1, 1, 1, layer4_bin[id_layer4_bin++], true);
        
        tk::dnn::Activation *relu1_0_2      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        tk::dnn::Conv2d     *layer1_0_conv3 = new tk::dnn::Conv2d(&net, 2048, 1, 1, 1, 1, 0, 0, layer4_bin[id_layer4_bin++], true);
        if(i==0) 
        {
            tk::dnn::Layer    *route_1_0_layers[1] = { last };
            tk::dnn::Route    *route_1_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
            tk::dnn::Conv2d   *layer1_0_downsample_0 = new tk::dnn::Conv2d(&net, 2048, 1, 1, 2, 2, 0, 0, layer4_bin[id_layer4_bin++], true);
            tk::dnn::Shortcut *s1_0                  = new tk::dnn::Shortcut(&net, layer1_0_conv3);
        } 
        else 
        {
            tk::dnn::Shortcut *s1_0                  = new tk::dnn::Shortcut(&net, last);
        }
        tk::dnn::Activation   *layer1_0_relu = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
        last = layer1_0_relu;
    }

    //final
    tk::dnn::Pooling avgpool(&net, 7, 7, 7, 7, 0, 0, tk::dnn::POOLING_AVERAGE);
    tk::dnn::Dense   fc(&net, 1000, fc_bin);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    //printDeviceVector(64, data, true);

    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("resnet101"));

    
    tk::dnn::dataDim_t out_dim;
    out_dim = net.layers[net.num_layers-1]->output_dim;
    dnnType *cudnn_out, *rt_out;

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TKDNN_TSTART
        net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }
    cudnn_out = net.layers[net.num_layers-1]->dstData;

    //printDeviceVector(64, cudnn_out, true);

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TKDNN_TSTART
        netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }
    rt_out = (dnnType *)netRT.buffersRT[1];


    printCenteredTitle(std::string(" RESNET CHECK RESULTS ").c_str(), '=', 30);
    dnnType *out, *out_h;
    int odim = out_dim.tot();
    readBinaryFile(output_bin, odim, &out_h, &out);
    
    std::cout<<"CUDNN vs correct"; 
    int ret_cudnn = checkResult(odim, cudnn_out, out) == 0 ? 0: ERROR_CUDNN;
    std::cout<<"TRT   vs correct"; 
    int ret_tensorrt = checkResult(odim, rt_out, out) == 0 ? 0 : ERROR_TENSORRT;
    std::cout<<"CUDNN vs TRT    "; 
    int ret_cudnn_tensorrt = checkResult(odim, cudnn_out, rt_out) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
