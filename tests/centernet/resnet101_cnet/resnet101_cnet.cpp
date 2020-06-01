#include <iostream>

#include "kernels.h"
#include "Yolo3Detection.h"
#include "tkdnn.h"
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort
// #include "utils.h"

const char *input_bin = "resnet101_cnet/debug/input.bin";
const char *conv1_bin = "resnet101_cnet/layers/conv1.bin";

//layer1
const char *layer1_bin[]={
"resnet101_cnet/layers/layer1-0-conv1.bin",
"resnet101_cnet/layers/layer1-0-conv2.bin",
"resnet101_cnet/layers/layer1-0-conv3.bin",
"resnet101_cnet/layers/layer1-0-downsample-0.bin",

"resnet101_cnet/layers/layer1-1-conv1.bin",
"resnet101_cnet/layers/layer1-1-conv2.bin",
"resnet101_cnet/layers/layer1-1-conv3.bin",

"resnet101_cnet/layers/layer1-2-conv1.bin",
"resnet101_cnet/layers/layer1-2-conv2.bin",
"resnet101_cnet/layers/layer1-2-conv3.bin"};
                        

//layer2
const char *layer2_bin[]={
"resnet101_cnet/layers/layer2-0-conv1.bin",
"resnet101_cnet/layers/layer2-0-conv2.bin",
"resnet101_cnet/layers/layer2-0-conv3.bin",
"resnet101_cnet/layers/layer2-0-downsample-0.bin",

"resnet101_cnet/layers/layer2-1-conv1.bin",
"resnet101_cnet/layers/layer2-1-conv2.bin",
"resnet101_cnet/layers/layer2-1-conv3.bin",

"resnet101_cnet/layers/layer2-2-conv1.bin",
"resnet101_cnet/layers/layer2-2-conv2.bin",
"resnet101_cnet/layers/layer2-2-conv3.bin",

"resnet101_cnet/layers/layer2-3-conv1.bin",
"resnet101_cnet/layers/layer2-3-conv2.bin",
"resnet101_cnet/layers/layer2-3-conv3.bin"
};
//layer3
const char *layer3_bin[]={
"resnet101_cnet/layers/layer3-0-conv1.bin",
"resnet101_cnet/layers/layer3-0-conv2.bin",
"resnet101_cnet/layers/layer3-0-conv3.bin",
"resnet101_cnet/layers/layer3-0-downsample-0.bin",

"resnet101_cnet/layers/layer3-1-conv1.bin",
"resnet101_cnet/layers/layer3-1-conv2.bin",
"resnet101_cnet/layers/layer3-1-conv3.bin",

"resnet101_cnet/layers/layer3-2-conv1.bin",
"resnet101_cnet/layers/layer3-2-conv2.bin",
"resnet101_cnet/layers/layer3-2-conv3.bin",

"resnet101_cnet/layers/layer3-3-conv1.bin",
"resnet101_cnet/layers/layer3-3-conv2.bin",
"resnet101_cnet/layers/layer3-3-conv3.bin",

"resnet101_cnet/layers/layer3-4-conv1.bin",
"resnet101_cnet/layers/layer3-4-conv2.bin",
"resnet101_cnet/layers/layer3-4-conv3.bin",

"resnet101_cnet/layers/layer3-5-conv1.bin",
"resnet101_cnet/layers/layer3-5-conv2.bin",
"resnet101_cnet/layers/layer3-5-conv3.bin",

"resnet101_cnet/layers/layer3-6-conv1.bin",
"resnet101_cnet/layers/layer3-6-conv2.bin",
"resnet101_cnet/layers/layer3-6-conv3.bin",

"resnet101_cnet/layers/layer3-7-conv1.bin",
"resnet101_cnet/layers/layer3-7-conv2.bin",
"resnet101_cnet/layers/layer3-7-conv3.bin",

"resnet101_cnet/layers/layer3-8-conv1.bin",
"resnet101_cnet/layers/layer3-8-conv2.bin",
"resnet101_cnet/layers/layer3-8-conv3.bin",

"resnet101_cnet/layers/layer3-9-conv1.bin",
"resnet101_cnet/layers/layer3-9-conv2.bin",
"resnet101_cnet/layers/layer3-9-conv3.bin",

"resnet101_cnet/layers/layer3-10-conv1.bin",
"resnet101_cnet/layers/layer3-10-conv2.bin",
"resnet101_cnet/layers/layer3-10-conv3.bin",

"resnet101_cnet/layers/layer3-11-conv1.bin",
"resnet101_cnet/layers/layer3-11-conv2.bin",
"resnet101_cnet/layers/layer3-11-conv3.bin",

"resnet101_cnet/layers/layer3-12-conv1.bin",
"resnet101_cnet/layers/layer3-12-conv2.bin",
"resnet101_cnet/layers/layer3-12-conv3.bin",

"resnet101_cnet/layers/layer3-13-conv1.bin",
"resnet101_cnet/layers/layer3-13-conv2.bin",
"resnet101_cnet/layers/layer3-13-conv3.bin",

"resnet101_cnet/layers/layer3-14-conv1.bin",
"resnet101_cnet/layers/layer3-14-conv2.bin",
"resnet101_cnet/layers/layer3-14-conv3.bin",

"resnet101_cnet/layers/layer3-15-conv1.bin",
"resnet101_cnet/layers/layer3-15-conv2.bin",
"resnet101_cnet/layers/layer3-15-conv3.bin",

"resnet101_cnet/layers/layer3-16-conv1.bin",
"resnet101_cnet/layers/layer3-16-conv2.bin",
"resnet101_cnet/layers/layer3-16-conv3.bin",

"resnet101_cnet/layers/layer3-17-conv1.bin",
"resnet101_cnet/layers/layer3-17-conv2.bin",
"resnet101_cnet/layers/layer3-17-conv3.bin",

"resnet101_cnet/layers/layer3-18-conv1.bin",
"resnet101_cnet/layers/layer3-18-conv2.bin",
"resnet101_cnet/layers/layer3-18-conv3.bin",

"resnet101_cnet/layers/layer3-19-conv1.bin",
"resnet101_cnet/layers/layer3-19-conv2.bin",
"resnet101_cnet/layers/layer3-19-conv3.bin",

"resnet101_cnet/layers/layer3-20-conv1.bin",
"resnet101_cnet/layers/layer3-20-conv2.bin",
"resnet101_cnet/layers/layer3-20-conv3.bin",

"resnet101_cnet/layers/layer3-21-conv1.bin",
"resnet101_cnet/layers/layer3-21-conv2.bin",
"resnet101_cnet/layers/layer3-21-conv3.bin",

"resnet101_cnet/layers/layer3-22-conv1.bin",
"resnet101_cnet/layers/layer3-22-conv2.bin",
"resnet101_cnet/layers/layer3-22-conv3.bin"};


//layer4
const char *layer4_bin[]={
"resnet101_cnet/layers/layer4-0-conv1.bin",
"resnet101_cnet/layers/layer4-0-conv2.bin",
"resnet101_cnet/layers/layer4-0-conv3.bin",
"resnet101_cnet/layers/layer4-0-downsample-0.bin",

"resnet101_cnet/layers/layer4-1-conv1.bin",
"resnet101_cnet/layers/layer4-1-conv2.bin",
"resnet101_cnet/layers/layer4-1-conv3.bin",

"resnet101_cnet/layers/layer4-2-conv1.bin",
"resnet101_cnet/layers/layer4-2-conv2.bin",
"resnet101_cnet/layers/layer4-2-conv3.bin"};

const char *d_conv1_bin = "resnet101_cnet/layers/deconv_layers-0-conv_offset_mask.bin";
const char *deform1_bin = "resnet101_cnet/layers/deconv_layers-0.bin";
const char *deconv1_bin = "resnet101_cnet/layers/deconv_layers-3.bin";

const char *d_conv2_bin = "resnet101_cnet/layers/deconv_layers-6-conv_offset_mask.bin";
const char *deform2_bin = "resnet101_cnet/layers/deconv_layers-6.bin";
const char *deconv2_bin = "resnet101_cnet/layers/deconv_layers-9.bin";

const char *d_conv3_bin = "resnet101_cnet/layers/deconv_layers-12-conv_offset_mask.bin";
const char *deform3_bin = "resnet101_cnet/layers/deconv_layers-12.bin";
const char *deconv3_bin = "resnet101_cnet/layers/deconv_layers-15.bin";

const char *hm_conv1_bin = "resnet101_cnet/layers/hm-0.bin";
const char *hm_conv2_bin = "resnet101_cnet/layers/hm-2.bin";
const char *wh_conv1_bin = "resnet101_cnet/layers/wh-0.bin";
const char *wh_conv2_bin = "resnet101_cnet/layers/wh-2.bin";
const char *reg_conv1_bin = "resnet101_cnet/layers/reg-0.bin";
const char *reg_conv2_bin = "resnet101_cnet/layers/reg-2.bin";
//final
const char *fc_bin = "resnet101_cnet/layers/fc.bin";

const char *output_bin[]={
"resnet101_cnet/debug/hm.bin",
"resnet101_cnet/debug/wh.bin",
"resnet101_cnet/debug/reg.bin"};

int main()
{
    downloadWeightsifDoNotExist(input_bin, "resnet101_cnet", "https://cloud.hipert.unimore.it/s/5BTjHMWBcJk8g3i/download");

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 512, 512, 1);
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

    tk::dnn::DeformConv2d     *layer0_deform1 = new tk::dnn::DeformConv2d(&net, 256, 1, 3, 3, 1, 1, 1, 1, deform1_bin, d_conv1_bin, true);
    tk::dnn::Activation   *layer0_deform1_relu = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d          *layer0_deconv1 = new tk::dnn::DeConv2d(&net, 256, 4, 4, 2, 2, 1, 1, deconv1_bin, true);
    tk::dnn::Activation *layer0_deconv1_relu      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::DeformConv2d     *layer1_deform1 = new tk::dnn::DeformConv2d(&net, 128, 1, 3, 3, 1, 1, 1, 1, deform2_bin, d_conv2_bin, true);
    tk::dnn::Activation   *layer1_deform1_relu = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d          *layer1_deconv1 = new tk::dnn::DeConv2d(&net, 128, 4, 4, 2, 2, 1, 1, deconv2_bin, true);
    tk::dnn::Activation *layer1_deconv1_relu      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    
    tk::dnn::DeformConv2d     *layer2_deform1 = new tk::dnn::DeformConv2d(&net, 64, 1, 3, 3, 1, 1, 1, 1, deform3_bin, d_conv3_bin, true);
    tk::dnn::Activation   *layer2_deform1_relu = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d          *layer2_deconv1 = new tk::dnn::DeConv2d(&net, 64, 4, 4, 2, 2, 1, 1, deconv3_bin, true);
    tk::dnn::Activation *layer2_deconv1_relu      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);          
        
    tk::dnn::Layer    *route_1_0_layers[1] = { layer2_deconv1_relu };
    tk::dnn::Conv2d     *hm_conv1 = new tk::dnn::Conv2d(&net, 64, 3, 3, 1, 1, 1, 1, hm_conv1_bin, false);
    tk::dnn::Activation *hm_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *hm = new tk::dnn::Conv2d(&net, 80, 1, 1, 1, 1, 0, 0, hm_conv2_bin, false);
    hm->setFinal();
    int kernel = 3; 
    int pad = (kernel - 1)/2;
    tk::dnn::Activation *hm_sig      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_SIGMOID);
    tk::dnn::Pooling  *hmax                 = new tk::dnn::Pooling(&net, kernel, kernel, 1, 1, pad, pad, tk::dnn::POOLING_MAX);
    hmax->setFinal();

    tk::dnn::Route    *route_1_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *wh_conv1 = new tk::dnn::Conv2d(&net, 64, 3, 3, 1, 1, 1, 1, wh_conv1_bin, false);
    tk::dnn::Activation *wh_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *wh = new tk::dnn::Conv2d(&net, 2, 1, 1, 1, 1, 0, 0, wh_conv2_bin, false);
    wh->setFinal();        
    
    tk::dnn::Route    *route_2_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *reg_conv1 = new tk::dnn::Conv2d(&net, 64, 3, 3, 1, 1, 1, 1, reg_conv1_bin, false);
    tk::dnn::Activation *reg_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *reg = new tk::dnn::Conv2d(&net, 2, 1, 1, 1, 1, 0, 0, reg_conv2_bin, false);
    reg->setFinal();

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    // printDeviceVector(64, data, true);

    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("resnet101_cnet"));

    
    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TKDNN_TSTART
        net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }

    // printDeviceVector(64, cudnn_out, true);

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TKDNN_TSTART
        netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }

    tk::dnn::Layer *outs[3] = { hm, wh, reg }; 
    int out_count = 1;
    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0; 
    for(int i=0; i<3; i++) {
        printCenteredTitle((std::string(" RESNET CHECK RESULTS ") + std::to_string(i) + " ").c_str(), '=', 30);
        
        outs[i]->output_dim.print();
        
        dnnType *out, *out_h;
        int odim = outs[i]->output_dim.tot();
        readBinaryFile(output_bin[i], odim, &out_h, &out);
        // std::cout<<"OUTPUT BIN:\n";
        // printDeviceVector(odim, cudnn_out, true);
        // std::cout<<"FILE BIN:\n";
        // printDeviceVector(odim, out, true);

        dnnType *cudnn_out, *rt_out;
        cudnn_out = outs[i]->dstData;
        rt_out = (dnnType *)netRT.buffersRT[i+out_count];
        // there is the maxpool. It isn't an output but it is necessary for the process section
        if(i==0)
            out_count ++;

        std::cout<<"CUDNN vs correct"; 
        ret_cudnn |= checkResult(odim, cudnn_out, out) == 0 ? 0: ERROR_CUDNN;
        std::cout<<"TRT   vs correct"; 
        ret_tensorrt |= checkResult(odim, rt_out, out) == 0 ? 0 : ERROR_TENSORRT;
        std::cout<<"CUDNN vs TRT    "; 
        ret_cudnn_tensorrt |= checkResult(odim, cudnn_out, rt_out) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    }
    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
