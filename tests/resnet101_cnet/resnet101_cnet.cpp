#include <iostream>

#include "kernels.h"
#include "Yolo3Detection.h"
#include "tkdnn.h"
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort
// #include "utils.h"

const char *input_bin = "../tests/resnet101_cnet/debug/input.bin";
const char *conv1_bin = "../tests/resnet101_cnet/layers/conv1.bin";

//layer1
const char *layer1_bin[]={
"../tests/resnet101_cnet/layers/layer1-0-conv1.bin",
"../tests/resnet101_cnet/layers/layer1-0-conv2.bin",
"../tests/resnet101_cnet/layers/layer1-0-conv3.bin",
"../tests/resnet101_cnet/layers/layer1-0-downsample-0.bin",

"../tests/resnet101_cnet/layers/layer1-1-conv1.bin",
"../tests/resnet101_cnet/layers/layer1-1-conv2.bin",
"../tests/resnet101_cnet/layers/layer1-1-conv3.bin",

"../tests/resnet101_cnet/layers/layer1-2-conv1.bin",
"../tests/resnet101_cnet/layers/layer1-2-conv2.bin",
"../tests/resnet101_cnet/layers/layer1-2-conv3.bin"};
                        

//layer2
const char *layer2_bin[]={
"../tests/resnet101_cnet/layers/layer2-0-conv1.bin",
"../tests/resnet101_cnet/layers/layer2-0-conv2.bin",
"../tests/resnet101_cnet/layers/layer2-0-conv3.bin",
"../tests/resnet101_cnet/layers/layer2-0-downsample-0.bin",

"../tests/resnet101_cnet/layers/layer2-1-conv1.bin",
"../tests/resnet101_cnet/layers/layer2-1-conv2.bin",
"../tests/resnet101_cnet/layers/layer2-1-conv3.bin",

"../tests/resnet101_cnet/layers/layer2-2-conv1.bin",
"../tests/resnet101_cnet/layers/layer2-2-conv2.bin",
"../tests/resnet101_cnet/layers/layer2-2-conv3.bin",

"../tests/resnet101_cnet/layers/layer2-3-conv1.bin",
"../tests/resnet101_cnet/layers/layer2-3-conv2.bin",
"../tests/resnet101_cnet/layers/layer2-3-conv3.bin"
};
//layer3
const char *layer3_bin[]={
"../tests/resnet101_cnet/layers/layer3-0-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-0-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-0-conv3.bin",
"../tests/resnet101_cnet/layers/layer3-0-downsample-0.bin",

"../tests/resnet101_cnet/layers/layer3-1-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-1-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-1-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-2-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-2-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-2-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-3-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-3-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-3-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-4-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-4-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-4-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-5-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-5-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-5-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-6-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-6-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-6-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-7-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-7-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-7-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-8-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-8-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-8-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-9-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-9-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-9-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-10-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-10-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-10-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-11-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-11-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-11-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-12-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-12-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-12-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-13-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-13-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-13-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-14-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-14-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-14-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-15-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-15-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-15-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-16-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-16-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-16-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-17-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-17-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-17-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-18-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-18-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-18-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-19-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-19-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-19-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-20-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-20-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-20-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-21-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-21-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-21-conv3.bin",

"../tests/resnet101_cnet/layers/layer3-22-conv1.bin",
"../tests/resnet101_cnet/layers/layer3-22-conv2.bin",
"../tests/resnet101_cnet/layers/layer3-22-conv3.bin"};


//layer4
const char *layer4_bin[]={
"../tests/resnet101_cnet/layers/layer4-0-conv1.bin",
"../tests/resnet101_cnet/layers/layer4-0-conv2.bin",
"../tests/resnet101_cnet/layers/layer4-0-conv3.bin",
"../tests/resnet101_cnet/layers/layer4-0-downsample-0.bin",

"../tests/resnet101_cnet/layers/layer4-1-conv1.bin",
"../tests/resnet101_cnet/layers/layer4-1-conv2.bin",
"../tests/resnet101_cnet/layers/layer4-1-conv3.bin",

"../tests/resnet101_cnet/layers/layer4-2-conv1.bin",
"../tests/resnet101_cnet/layers/layer4-2-conv2.bin",
"../tests/resnet101_cnet/layers/layer4-2-conv3.bin"};

const char *d_conv1_bin = "../tests/resnet101_cnet/layers/deconv_layers-0-conv_offset_mask.bin";
const char *deform1_bin = "../tests/resnet101_cnet/layers/deconv_layers-0.bin";
const char *deconv1_bin = "../tests/resnet101_cnet/layers/deconv_layers-3.bin";

const char *d_conv2_bin = "../tests/resnet101_cnet/layers/deconv_layers-6-conv_offset_mask.bin";
const char *deform2_bin = "../tests/resnet101_cnet/layers/deconv_layers-6.bin";
const char *deconv2_bin = "../tests/resnet101_cnet/layers/deconv_layers-9.bin";

const char *d_conv3_bin = "../tests/resnet101_cnet/layers/deconv_layers-12-conv_offset_mask.bin";
const char *deform3_bin = "../tests/resnet101_cnet/layers/deconv_layers-12.bin";
const char *deconv3_bin = "../tests/resnet101_cnet/layers/deconv_layers-15.bin";

const char *hm_conv1_bin = "../tests/resnet101_cnet/layers/hm-0.bin";
const char *hm_conv2_bin = "../tests/resnet101_cnet/layers/hm-2.bin";
const char *wh_conv1_bin = "../tests/resnet101_cnet/layers/wh-0.bin";
const char *wh_conv2_bin = "../tests/resnet101_cnet/layers/wh-2.bin";
const char *reg_conv1_bin = "../tests/resnet101_cnet/layers/reg-0.bin";
const char *reg_conv2_bin = "../tests/resnet101_cnet/layers/reg-2.bin";
//final
const char *fc_bin = "../tests/resnet101_cnet/layers/fc.bin";

const char *output_bin[]={
"../tests/resnet101_cnet/debug/hm.bin",
"../tests/resnet101_cnet/debug/wh.bin",
"../tests/resnet101_cnet/debug/reg.bin"};



std::vector<size_t> sort_indexes(const std::vector<float> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  return idx;
}

float _colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * _colors[i % 6][c % 3] + ratio*_colors[j % 6][c % 3];
    //printf("%f\n", r);
    return r;
}

int computeDetections(dnnType *hm_d, dnnType *wh_d, dnnType *reg_d, int hm_dim, int wh_dim, int reg_dim, bool cat_spec_wh, int k){
    // _nms
    int kernel = 3; 
    int pad = (kernel - 1)/2;
    std::cout<<"computeDetections\n";
    // dnnType *hmax;
    // tk::dnn::Pooling maxpool(&hmax, 3, 3, 2, 2, 1, 1, tk::dnn::POOLING_MAX)
    // =  (dnnType *) 
    
    // net.functional.max_pool2d(
    //     heat, (kernel, kernel), stride=1, padding=pad)
    // keep = (hmax == heat).float()
    // return heat * keep
}

int process(dnnType *hm_d, dnnType *wh_d, dnnType *reg_d, int hm_dim, int wh_dim, int reg_dim){
    std::cout<<"process\n";
    // computeDetections(hm_d, wh_d, reg_d, hm_dim, wh_dim, reg_dim, false, 100);
}

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
    tk::dnn::Conv2d     *hm = new tk::dnn::Conv2d(&net, 80, 1, 1, 1, 1, 0, 0, hm_conv2_bin, false, false, true);
    int kernel = 3; 
    int pad = (kernel - 1)/2;
    tk::dnn::Activation *hm_sig      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_SIGMOID);
    tk::dnn::Pooling  *hmax                 = new tk::dnn::Pooling(&net, kernel, kernel, 1, 1, pad, pad, tk::dnn::POOLING_MAX, true);

    tk::dnn::Route    *route_1_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *wh_conv1 = new tk::dnn::Conv2d(&net, 64, 3, 3, 1, 1, 1, 1, wh_conv1_bin, false);
    tk::dnn::Activation *wh_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *wh = new tk::dnn::Conv2d(&net, 2, 1, 1, 1, 1, 0, 0, wh_conv2_bin, false, false, true);        
    
    tk::dnn::Route    *route_2_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *reg_conv1 = new tk::dnn::Conv2d(&net, 64, 3, 3, 1, 1, 1, 1, reg_conv1_bin, false);
    tk::dnn::Activation *reg_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *reg = new tk::dnn::Conv2d(&net, 2, 1, 1, 1, 1, 0, 0, reg_conv2_bin, false, false, true);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    // printDeviceVector(64, data, true);

    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, "resnet101_cnet.rt");

    
    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TIMER_START
        net.infer(dim1, data);
        TIMER_STOP
        dim1.print();
    }

    // printDeviceVector(64, cudnn_out, true);

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TIMER_START
        netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }

    tk::dnn::Layer *outs[3] = { hm, wh, reg }; 
    int out_count = 1;
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

        std::cout << "CUDNN vs correct";
        checkResult(odim, cudnn_out, out);

        std::cout << "TRT   vs correct";
        checkResult(odim, rt_out, out);
        std::cout << "CUDNN vs TRT    ";
        checkResult(odim, cudnn_out, rt_out);
    }

    TIMER_START

    // -------- transofrm compose
    cv::Mat imageOrig = cv::imread("/media/davide/DATA/shared_home/Projects/Professionale/repos/photo_2020-01-14_09-56-07.jpg");
    cv::Mat imageF;
    imageOrig.convertTo(imageF, CV_32FC3, 1/255.0); 
    cv::Mat image;
    cv::Size sz = imageF.size();
    std::cout<<"image: "<<sz.width<<", "<<sz.height<<std::endl;
    resize(imageF, image, cv::Size(256, 256));
    const int cropSize = 224;
    const int offsetW = (image.cols - cropSize) / 2;
    const int offsetH = (image.rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
    image = image(roi).clone();
    std::cout << "Cropped image dimension: " << image.cols << " X " << image.rows << std::endl;

    cv::Scalar mean_; 
    mean_ << 0.485, 0.456, 0.406;
    cv::Scalar stddev_; 
    stddev_ << 0.229, 0.224, 0.225;
    cv::Size s_im = imageF.size();
    std::cout<<"size: "<<s_im.height<<" "<<s_im.width<<" - "<<std::endl;
    std::cout<<"mean: "<<mean_<<", std: "<<stddev_<<std::endl;
    cv::add(image, -mean_, image);
    cv::divide(image, stddev_, image);
    cv::Mat bgr[3]; 
    dnnType *input, *input_d;
    dim2 = dim;
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*dim2.tot()));
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*dim2.tot()));
    image.convertTo(image, CV_32FC3, 1/255.0); 

    //split channels
    cv::split(image,bgr);//split source

    //write channels
    for(int i=0; i<dim2.c; i++) {
        int idx = i*image.rows*image.cols;
        int ch = dim2.c-1 -i;
        memcpy((void*)&input[idx], (void*)bgr[ch].data, image.rows*image.cols*sizeof(dnnType));
    }

    checkCuda(cudaMemcpyAsync(input_d, input, dim2.tot()*sizeof(dnnType), cudaMemcpyHostToDevice));

    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        netRT.infer(dim2, input_d);
        TIMER_STOP
        dim2.print();
    }
    checkResult(dim2.tot(), input_h, input);
    checkCuda(cudaFree(input_d));
    checkCuda(cudaFreeHost(input));
    // -----------------------------------pre-process ------------------------------------------
    // it will resize the images to `512 x 512` in GETTING_STARTED.md
    
    float scale = 1.0;
    float new_height = sz.height * scale;
    float new_width = sz.width * scale;
    float inp_height = 224;//512;
    float inp_width = 224;//512;
    float c[] = {new_width / 2.0, new_height /2.0};
    float s[2];
    if(sz.width > sz.height){
        s[0] = sz.width * 1.0;
        s[1] = sz.width * 1.0;
    }
    else{
        s[0] = sz.height * 1.0;    
        s[1] = sz.height * 1.0;    
    }
    std::cout<<" "<<new_height<<" "<<new_width<<" "<<s[0]<<"-"<<s[1]<<" "<<c[0]<<"-"<<c[1]<<std::endl; 
    // ----------- get_affine_transform
    // rot_rad = pi * 0 / 100 --> 0
    cv::Mat src(cv::Size(2,3), CV_32F);
    cv::Mat dst(cv::Size(2,3), CV_32F);    
    src.at<float>(0,0)=c[0];
    src.at<float>(0,1)=c[1];
    src.at<float>(1,0)=c[0];
    src.at<float>(1,1)=c[1] + s[0] * -0.5;
    dst.at<float>(0,0)=inp_width * 0.5;
    dst.at<float>(0,1)=inp_height * 0.5;
    dst.at<float>(1,0)=inp_width * 0.5;
    dst.at<float>(1,1)=inp_height * 0.5 +  inp_width * -0.5; 
    
    src.at<float>(2,0)=src.at<float>(1,0) + (-src.at<float>(0,1)+src.at<float>(1,1) );
    src.at<float>(2,1)=src.at<float>(1,1) + (src.at<float>(0,0)-src.at<float>(1,0) );
    dst.at<float>(2,0)=dst.at<float>(1,0) + (-dst.at<float>(0,1)+dst.at<float>(1,1) );
    dst.at<float>(2,1)=dst.at<float>(1,1) + (dst.at<float>(0,0)-dst.at<float>(1,0) );
    std::cout<<"src: "<<src<<std::endl;
    std::cout<<"dst: "<<dst<<std::endl;

    cv::Mat trans = cv::getAffineTransform( src, dst );

    resize(imageOrig, image, cv::Size(new_width, new_height));
    s_im = image.size();
    std::cout<<"size: "<<s_im.height<<" "<<s_im.width<<" - "<<std::endl;
    

    // image.convertTo(image, CV_32FC3, 1/255.0); 
    cv::warpAffine(image, image, trans, cv::Size(inp_width, inp_height), cv::INTER_LINEAR );
    // cv::Scalar mean, stddev;
    s_im = image.size();
    std::cout<<"size: "<<s_im.height<<" "<<s_im.width<<" - "<<std::endl;
    
    // for(int i=0; i<dim.tot(); i++ ){
    //     std::cout<<(float)image.at<cv::Vec3b>(0,i)[0]<<" -  "<<(float)image.at<cv::Vec3b>(0,i)[1]<<" - "<<(float)image.at<cv::Vec3b>(0,i)[2]<<" - "<<std::endl;
    //     if(i==10)
    //         break;
    // }
    // return 0;
    /////////////////////////////// ok fin qui


    // cv::meanStdDev(image, mean, stddev );
    // cv::Scalar mean(0.408, 0.447, 0.47); 
    cv::Vec<float, 3> mean;
    mean << 0.408, 0.447, 0.47;
    // s_im = mean.size();
    // std::cout<<"size: "<<s_im.height<<" "<<s_im.width<<" - "<<std::endl;
    
    cv::Vec<float, 3> stddev; 
    stddev << 0.289, 0.274, 0.278;

    cv::Size s_imag = image.size();
    std::cout<<"size: "<<s_imag.height<<" "<<s_imag.width<<" - "<<std::endl;
    image.convertTo(image, CV_32FC3, 1/255.0); 
    
    std::cout<<"mean: "<<mean<<", std: "<<stddev<<std::endl;
    // cv::add(image, -mean, image);
    // cv::divide(image, stddev, image);
    cv::MatIterator_<cv::Vec<float, 3>> it;
    for(it = image.begin<cv::Vec<float, 3>>(); it != image.end<cv::Vec<float, 3>>(); ++it)
    {
        (*it)[0] = (float)(*it)[0] - mean[0];
        (*it)[1] = (float)(*it)[1] - mean[1];
        (*it)[2] = (float)(*it)[2] - mean[2];
        (*it)[0] = (float)(*it)[0] / stddev[0];
        (*it)[1] = (float)(*it)[1] / stddev[1];
        (*it)[2] = (float)(*it)[2] / stddev[2];
    }

    // for(int i=0; i<dim.tot(); i++ ){
    //     std::cout<<(float)image.at<cv::Vec<float, 3>>(0,i)[0]<<" -  "<<(float)image.at<cv::Vec<float, 3>>(0,i)[1]<<" - "<<(float)image.at<cv::Vec<float, 3>>(0,i)[2]<<" - "<<std::endl;
    //     if(i==10)
    //         break;
    // }
    // return 0;
    /////////////////////// ok fin qui
    
    
    std::cout<<"size: "<<s_imag.height<<" "<<s_imag.width<<" - "<<std::endl;

    cv::Mat bgr2[3]; 
    dnnType *input2, *input_d2;
    dim2 = dim;
    checkCuda(cudaMalloc(&input_d2, sizeof(dnnType)*dim2.tot()));
    checkCuda(cudaMallocHost(&input2, sizeof(dnnType)*dim2.tot()));
    // image.convertTo(image, CV_32FC3, 1/255.0); 

    //split channels
    cv::split(image,bgr2);//split source
    
    // std::cout<<"ch:\n";
    // for(int i=0; i<dim.tot(); i++ ){
    //     std::cout<<(float)bgr2[0].at<cv::Vec<float, 3>>(0,i)[0]<<" - "<<(float)bgr2[0].at<cv::Vec<float, 3>>(0,i)[1]<<" - "<<(float)bgr2[0].at<cv::Vec<float, 3>>(0,i)[2]<<std::endl;
    //     if(i==10)
    //         break;
    // }
    // std::cout<<"ch:\n";
    // for(int i=0; i<dim.tot(); i++ ){
    //     std::cout<<(float)bgr2[1].at<float>(1,i)<<std::endl;
    //     if(i==10)
    //         break;
    // }
    // std::cout<<"ch:\n";
    // for(int i=0; i<dim.tot(); i++ ){
    //     std::cout<<(float)bgr2[2].at<float>(2,i)<<std::endl;
    //     if(i==10)
    //         break;
    // }
    // return 0;
    ///////////////////// ok fin qui
    std::cout<<"\n\n\ncome: \n"<<image.rows<<" - "<<image.cols<<std::endl;
    std::cout<<"reprint shape dim2\n";
    dim2.print();
    std::cout<<std::endl;
    //write channels
    for(int i=0; i<dim2.c; i++) {
        int idx = i*image.rows*image.cols;
        int ch = dim2.c-3 +i;
        std::cout<<"i: "<<i<<", idx: "<<idx<<", ch: "<<ch<<std::endl;
        memcpy((void*)&input2[idx], (void*)bgr2[ch].data, image.rows*image.cols*sizeof(dnnType));
    }
    // int k100 = 0;
    // for(int i=1; i<=image.rows*image.cols*dim2.c; i++ ){
    //     std::cout<<input2[i]<<" ";
    //     if (i % (image.rows*image.cols) == 0){
    //         std::cout<<"\n\n";
    //         k100 ++;
    //     }
            
    // }
    // std::cout<<std::endl;
    // std::cout<<"ci sono "<<k100<<" r\n";
    // return 0;
    ///////////////////// pseudo ok

    checkCuda(cudaMemcpyAsync(input_d2, input2, dim2.tot()*sizeof(dnnType), cudaMemcpyHostToDevice));

    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        netRT.infer(dim2, input_d2);
        TIMER_STOP
        dim2.print();
    }
    checkResult(dim2.tot(), input_h, input2);
    for(int i=0; i<dim.tot(); i++ ){
        std::cout<<input_h[i]<<" "<<input2[i]<<std::endl;
        if(i==10)
            break;
    }
    // for(int i=0; i<dim.tot(); i++ ){
    //     std::cout<<input2[i]<<" ";
    // }
    // std::cout<<std::endl;
    // return 0;
    checkCuda(cudaFree(input_d2));
    checkCuda(cudaFreeHost(input2));
    // ------------------------------------ process --------------------------------------------
    dnnType *hm_h;
    checkCuda( cudaMallocHost(&hm_h, hm->output_dim.tot()*sizeof(dnnType)) );

    dnnType *rt_out[4];
    rt_out[0] = (dnnType *)netRT.buffersRT[1];
    rt_out[1] = (dnnType *)netRT.buffersRT[2];
    rt_out[2] = (dnnType *)netRT.buffersRT[3]; 
    rt_out[3] = (dnnType *)netRT.buffersRT[4]; 
    
    // checkCuda( cudaMemcpy(hm_h, rt_out[0], hm->output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );
    std::cout<<"hm\n";
    hm->output_dim.print();
    // for(int i=0; i<hm->output_dim.tot(); i++ ){
    //     std::cout<<hm_h[i]<<" ";
    //     if(i==100)
    //         break;
    // }
    // std::cout<<"\n";


    activationSIGMOIDForward(rt_out[0], rt_out[0], hm->output_dim.tot());
    checkCuda( cudaDeviceSynchronize() );
    
    
    checkCuda( cudaMemcpy(hm_h, rt_out[0], hm->output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );
    std::cout<<"hm\n";
    hm->output_dim.print();
    // for(int i=0; i<hm->output_dim.tot(); i++ ){
    //     std::cout<<hm_h[i]<<" ";
    //     if(i==100)
    //         break;
    // }
    // std::cout<<"\n";
    // return 0;
    //////////////////////////// ok
    // ----------- ctdet_decode
    // perform nms on heatmaps
    // tk::dnn::Layer    *route_hm_layers[1]   = { hm };
    // tk::dnn::Route    *route_hm             = new tk::dnn::Route(&net, route_hm_layers, 1);
    
    // ----------- nms
    // int kernel = 3; 
    // int pad = (kernel - 1)/2;
    // tk::dnn::Pooling  *hmax                 = new tk::dnn::Pooling(&net, kernel, kernel, 1, 1, pad, pad, tk::dnn::POOLING_MAX);
    
    // hmax_d = hmax->infer(hmax->input_dim.tot(), rt_out[0]);
    // keep = (hmax == heat).float()
    // return heat * keep
    
    dnnType *hmax_h;
    checkCuda( cudaMallocHost(&hmax_h, hmax->output_dim.tot()*sizeof(dnnType)) );
    checkCuda( cudaMemcpy(hmax_h, rt_out[1], hmax->output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );

    std::cout<<"hmax\n";
    hmax->output_dim.print();
    // for(int i=0; i<hmax->output_dim.tot(); i++ ){
    //     std::cout<<hmax_h[i]<<" ";
    //     if(i==100)
    //         break;
    // }
    // std::cout<<"\n";
    // return 0;
    // hm = hm * ( hmax == hm );
    std::cout<<"hm:\n";
    hm->output_dim.print();
    std::cout<<"hmax:\n";
    hmax->output_dim.print();
    // return 0;
    float toll = 0.000001;
    for(int i=0; i < hm->output_dim.tot(); i++){
        if(hm_h[i]-hmax_h[i] > toll || hm_h[i]-hmax_h[i] < -toll){
            hm_h[i] = 0.0f;
        }
    }
    // std::cout<<"\n";
    // for(int i=0; i<hm->output_dim.tot(); i++ ){
    //     std::cout<<hm_h[i]<<" ";
    //     if(i==100)
    //         break;
    // }
    // std::cout<<"\n";
    // return 0;
    // checkCuda( cudaMemcpy(hm->dstData, hm_h, hm->output_dim.tot()*sizeof(dnnType), cudaMemcpyHostToDevice) );
    checkCuda( cudaFreeHost(hmax_h) );
    // ----------- nms end
    // ----------- topk
    int K = 100;
    int width = 56;        // TODO
    float *topk_scores;
    int *topk_inds_;
    float *topk_ys_;
    float *topk_xs_;
    std::cout<<"mah: "<<hm->output_dim.c * K<<std::endl;
    checkCuda( cudaMallocHost(&topk_scores, hm->output_dim.c * K *sizeof(float)) );
    checkCuda( cudaMallocHost(&topk_inds_, hm->output_dim.c * K *sizeof(int)) );      
    checkCuda( cudaMallocHost(&topk_ys_, hm->output_dim.c * K *sizeof(float)) );      
    checkCuda( cudaMallocHost(&topk_xs_, hm->output_dim.c * K *sizeof(float)) );      
    std::cout<<"1\n";
    dnnType *hm_aus;
    checkCuda( cudaMallocHost(&hm_aus, hm->output_dim.h * hm->output_dim.w *sizeof(dnnType)) );     
    std::cout<<"2\n";
    int count;
    std::vector<float> v = {2.0, 3.0, 9.0};
    for (auto i: sort_indexes(v)) {
        std::cout << i<< "--" <<v[i] << std::endl;
    }
    for(int i=0; i<hm->output_dim.c; i++){
        count = 0;
        // get the hm->output_dim.h * hm->output_dim.w elements for each channel and sort it. Then find the first 100 elements
        checkCuda( cudaMemcpy(hm_aus, hm_h + i * hm->output_dim.h * hm->output_dim.w, 
                    hm->output_dim.h * hm->output_dim.w * sizeof(dnnType), cudaMemcpyHostToHost) );
        // std::cout<<"top scores: "<<hm->output_dim.h * hm->output_dim.w<<"\n";
        // for(int k=0; k<hm->output_dim.h * hm->output_dim.w; k++)
        //     std::cout<<hm_aus[k]<<" ";
        // std::cout<<std::endl;
        // std::vector<float> my_vector {arr, arr + arr_length}
        std::vector<float> my_vector{hm_aus, hm_aus +  hm->output_dim.h * hm->output_dim.w};
        for (auto j: sort_indexes(my_vector)) {
            // std::cout <<"j: "<<j<<" -> "<< hm_aus[j] << std::endl;
            topk_scores[i*K + count] = hm_aus[j];
            topk_inds_[i*K +count] = j;
            topk_ys_[i*K +count] = (int)(j / width);
            topk_xs_[i*K +count] = (int)(j % width);
            if(++count == K)
                break;
        }
    }
    std::cout<<"topk_xs_[0]: "<<topk_xs_[0]<<std::endl;
    for(int i = 0; i< hm->output_dim.c * K; i++)
        std::cout<<topk_xs_[i]<<" ";
    std::cout<<"\n3\n";
    // final
    float *scores;
    int *clses; 
    int *topk_inds;
    float *topk_ys; 
    float *topk_xs;
    checkCuda( cudaMallocHost(&scores, K *sizeof(float)) );
    checkCuda( cudaMallocHost(&clses, K *sizeof(int)) );
    checkCuda( cudaMallocHost(&topk_inds, K *sizeof(int)) );
    checkCuda( cudaMallocHost(&topk_ys, K *sizeof(float)) );     
    checkCuda( cudaMallocHost(&topk_xs, K *sizeof(float)) ); 
    std::cout<<"4\n";     
    count = 0;
    std::vector<float> my_vector{topk_scores, topk_scores +  hm->output_dim.c * K };
    for (auto j: sort_indexes(my_vector)) {
        // std::cout <<"j: "<<j<<" -> "<< hm_aus[j] << std::endl;
        scores[count] = topk_scores[j];
        clses[count] = (int)(j / K);
        topk_inds[count] = topk_inds_[j];
        topk_ys[count] = topk_ys_[j];
        topk_xs[count] = topk_xs_[j];
        if(++count == K)
            break;
    }
    checkCuda( cudaFreeHost(topk_scores) );
    checkCuda( cudaFreeHost(topk_inds_) );
    checkCuda( cudaFreeHost(topk_ys_) );
    checkCuda( cudaFreeHost(topk_xs_) );
    std::cout<<"5\n";
    // ----------- topk end 
    std::cout<<"topk_xs[0]: "<<topk_xs[0]<<std::endl;
    for(int i = 0; i< K; i++)
        std::cout<<topk_xs[i]<<" ";
    std::cout<<std::endl;
    /////////////////////////////// fin qui ok
    dnnType *reg_aus;
    checkCuda( cudaMallocHost(&reg_aus, reg->output_dim.tot()*sizeof(dnnType)) ); 
    checkCuda( cudaMemcpy(reg_aus, rt_out[3], reg->output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );
    std::cout<<"reg:\n";
    reg->output_dim.print();
    // for(int i=0; i<reg->output_dim.tot(); i++ ){
    //     std::cout<<reg_aus[i]<<" ";
    //     if(i==100)
    //         break;
    // }
    // std::cout<<"\n";
    // return 0;
    /////////////// ok    
    // for(int i=0; i<K; i++ ){
    //     std::cout<<reg_aus[topk_inds[i]]<<" "<<reg_aus[topk_inds[i]+56*56]<<std::endl;
    // }
    // std::cout<<"\n";
    // return 0;
    ///////////////////// ok fin qui
    

    for(int i = 0; i < K; i++){
        topk_xs[i] = topk_xs[i] + reg_aus[topk_inds[i]];
        topk_ys[i] = topk_ys[i] + reg_aus[topk_inds[i]+reg->output_dim.h*reg->output_dim.w];
    }
    std::cout<<"topk_xs[0]: "<<topk_xs[0]<<std::endl;
    checkCuda( cudaFreeHost(reg_aus) );
    std::cout<<"6\n";
    dnnType *wh_aus;
    float *bboxes;

    checkCuda( cudaMallocHost(&wh_aus, wh->output_dim.tot()*sizeof(dnnType)) ); 
    checkCuda( cudaMallocHost(&bboxes, 4 * K *sizeof(dnnType)) ); 
    checkCuda( cudaMemcpy(wh_aus, rt_out[2], wh->output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );    
    std::cout<<"7\n";
    for(int i = 0; i< K; i++)
        std::cout<<topk_xs[i]<<" ";
    std::cout<<std::endl;

    std::cout<<topk_xs[0]<<std::endl;
    std::cout<<topk_inds[0]<<std::endl;
    std::cout<<wh_aus[topk_inds[0]*2]<<std::endl;
    for(int i = 0; i < K; i++){
        bboxes[i * 4] = topk_xs[i] - wh_aus[topk_inds[i]] / 2;
        bboxes[i * 4 + 1] = topk_ys[i] - wh_aus[topk_inds[i]+reg->output_dim.h*reg->output_dim.w] / 2;
        bboxes[i * 4 + 2] = topk_xs[i] + wh_aus[topk_inds[i]] / 2;
        bboxes[i * 4 + 3] = topk_ys[i] + wh_aus[topk_inds[i]+reg->output_dim.h*reg->output_dim.w] / 2;
    }
    ////////////////// fin qui ok 

    checkCuda( cudaFreeHost(wh_aus) );
    checkCuda( cudaFreeHost(topk_inds) );
    checkCuda( cudaFreeHost(topk_ys) );
    checkCuda( cudaFreeHost(topk_xs) );

    std::cout<<"8\n";
    float *detections;
    std::cout<<"bboxes:\n";
    for(int i = 0; i < K+1; i++){
        std::cout<<bboxes[i]<<" ";
    }
    std::cout<<std::endl;
    checkCuda( cudaMallocHost(&detections, 6 * K *sizeof(dnnType)) ); 
    checkCuda( cudaMemcpy(detections, bboxes,  4 * K *sizeof(dnnType), cudaMemcpyHostToHost) );
    checkCuda( cudaMemcpy(detections + 4 * K *sizeof(dnnType), scores,  K *sizeof(dnnType), cudaMemcpyHostToHost) );
    checkCuda( cudaMemcpy(detections + 5 * K *sizeof(dnnType), clses,  K *sizeof(dnnType), cudaMemcpyHostToHost) );
    checkCuda( cudaFreeHost(bboxes) );
    // checkCuda( cudaFreeHost(scores) );
    // checkCuda( cudaFreeHost(clses) );
    // servono [bboxes, scores, clses]
    checkCuda( cudaDeviceSynchronize() );
    // ---------------------------------- post-process -----------------------------------------
    
    // --------- ctdet_post_process
    //for 1
    // float *dets;
    // checkCuda( cudaMallocHost(&dets, 2 * K *sizeof(float)) ); 
    // checkCuda( cudaMemcpy(dets, detections,  2 * K *sizeof(float), cudaMemcpyHostToHost) );
    // --------- transform_preds
    float *target_coords;
    checkCuda( cudaMallocHost(&target_coords, 4 * K *sizeof(float)) ); 
    src.at<float>(0,0)=c[0];
    src.at<float>(0,1)=c[1];
    src.at<float>(1,0)=c[0];
    src.at<float>(1,1)=c[1] + s[0] * -0.5;
    dst.at<float>(0,0)=width * 0.5;
    dst.at<float>(0,1)=width * 0.5;
    dst.at<float>(1,0)=width * 0.5;
    dst.at<float>(1,1)=width * 0.5 +  width * -0.5; 
    
    src.at<float>(2,0)=src.at<float>(1,0) + (-src.at<float>(0,1)+src.at<float>(1,1) );
    src.at<float>(2,1)=src.at<float>(1,1) + (src.at<float>(0,0)-src.at<float>(1,0) );
    dst.at<float>(2,0)=dst.at<float>(1,0) + (-dst.at<float>(0,1)+dst.at<float>(1,1) );
    dst.at<float>(2,1)=dst.at<float>(1,1) + (dst.at<float>(0,0)-dst.at<float>(1,0) );
    std::cout<<"src: "<<src<<std::endl;
    std::cout<<"dst: "<<dst<<std::endl;
    cv::Size s_aus;
    cv::Mat trans2(cv::Size(3,2), CV_32F);
    trans2 = cv::getAffineTransform( dst, src );
    s_aus = trans2.size();
    std::cout<<"trnas2: "<<trans2<<std::endl;
    std::cout<<trans2.at<double>(0,0)<<" - "<<trans2.at<double>(0,1)<<" - "<<trans2.at<double>(0,2)<<"\n"<<trans2.at<double>(1,0)<<" - "<<trans2.at<double>(1,1)<<" - "<<trans2.at<double>(1,2)<<std::endl;
    std::cout<<"size: "<<s_aus.height<<" "<<s_aus.width<<" - "<<std::endl;

    cv::Mat new_pt1(cv::Size(1,2), CV_32F);
    cv::Mat new_pt2(cv::Size(1,2), CV_32F);
    for(int i = 0; i<K; i++){
        // new_pt1.at<float>(0,0)=detections[i*4];
        // new_pt1.at<float>(0,1)=detections[i*4+1];
        // new_pt1.at<float>(0,2)=1.0;
        // new_pt1 << detections[i], detections[i+K], 1.0;
        // std::cout<<"----\ni: "<<i<<std::endl;//<<" newpt: "<<new_pt1<<std::endl;
        // std::cout<<"origi: "<<detections[i*4]<<", "<<detections[i*4+1]<<", "<<1.0<<std::endl;
        s_aus = new_pt1.size();
    
        // std::cout<<"size: "<<s_aus.height<<" "<<s_aus.width<<" - "<<std::endl;
        // new_pt2 = trans2.dot(new_pt1);
        new_pt1.at<float>(0,0)=static_cast<float>(trans2.at<double>(0,0))*detections[i*4] +
                                static_cast<float>(trans2.at<double>(0,1))*detections[i*4+1] +
                                static_cast<float>(trans2.at<double>(0,2))*1.0;
        new_pt1.at<float>(0,1)=static_cast<float>(trans2.at<double>(1,0))*detections[i*4] +
                                static_cast<float>(trans2.at<double>(1,1))*detections[i*4+1] +
                                static_cast<float>(trans2.at<double>(1,2))*1.0;

        new_pt2.at<float>(0,0)=static_cast<float>(trans2.at<double>(0,0))*detections[i*4+2] +
                                static_cast<float>(trans2.at<double>(0,1))*detections[i*4+3] +
                                static_cast<float>(trans2.at<double>(0,2))*1.0;
        new_pt2.at<float>(0,1)=static_cast<float>(trans2.at<double>(1,0))*detections[i*4+2] +
                                static_cast<float>(trans2.at<double>(1,1))*detections[i*4+3] +
                                static_cast<float>(trans2.at<double>(1,2))*1.0;

                                
        // std::cout<<"\n new: "<<new_pt1<<" - "<<new_pt2<<std::endl;
        target_coords[i*4] = new_pt1.at<float>(0,0);
        target_coords[i*4+1] = new_pt1.at<float>(0,1);
        target_coords[i*4+2] = new_pt2.at<float>(0,0);
        target_coords[i*4+3] = new_pt2.at<float>(0,1);
        // std::cout<<new_pt1.at<float>(0,0)<<", "<<new_pt1.at<float>(0,1)<<", "<<new_pt2.at<float>(0,0)<<", "<<new_pt2.at<float>(0,1)<<std::endl;

    }
    // return 0;
    // /////////////// ok fin qui
    const char *coco_class_name_ [] = {"person", "bicycle", "car", "motorcycle", "airplane", 
     "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
     "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
     "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
     "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
     "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
     "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
     "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
     "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
     "scissors", "teddy bear", "hair drier", "toothbrush"};

    std::vector<std::string> coco_class_name(coco_class_name_, std::end( coco_class_name_ ));
    int num_classes = 80;
    float vis_threshold = 0.3;
    // int *classes;
    std::vector<tk::dnn::box> detected;
    // checkCuda( cudaMallocHost(&classes, K *sizeof(int)) );
    // checkCuda( cudaMemcpy(classes, detections + 5 * K *sizeof(dnnType),  K *sizeof(dnnType), cudaMemcpyHostToHost) );
    for(int i = 0; i<num_classes; i++){
        for(int j=0; j<K; j++)
            if(clses[j] == i){
                //TODO recupera detections[j +0 +1 +2 +3 +4(+5 è la classes, già presa)]
                // queste compongono un ogg assegnato alla classe i (0, 79) --> i+1 (1:80);
                
                if(scores[j] > vis_threshold){
                    std::cout<<"th: "<<scores[j]<<" - cl: "<<clses[j]<<" i: "<<i<<std::endl;
                    //add coco bbox
                    //det[0:4], i, det[4]
                    int x0   = target_coords[j*4];
                    int y0   = target_coords[j*4+1];
                    int x1   = target_coords[j*4+2];
                    int y1   = target_coords[j*4+3];
                    int obj_class = clses[j];
                    float prob = scores[j];
                    std::cout<<"("<<x0<<", "<<y0<<"),("<<x1<<", "<<y1<<")"<<std::endl;
                    tk::dnn::box res;
                    res.cl = obj_class;
                    res.prob = prob;
                    res.x = x0;
                    res.y = y0;
                    res.w = x1 - x0;
                    res.h = y1 - y0;
                    detected.push_back(res);
                }
            }
    }
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;
    ;
    
    
    int baseline = 0;
    float fontScale = 0.5;
    int thickness = 2;
    cv::Scalar colors[256];
    for(int c=0; c<num_classes; c++) {
        int offset = c*123457 % num_classes;
        float r = get_color(2, offset, num_classes);
        float g = get_color(1, offset, num_classes);
        float b = get_color(0, offset, num_classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }
    int num_detected = detected.size();
    for (int i = 0; i < num_detected; i++){
        b = detected[i];
        x0 = b.x;
        w = b.w;
        x1 = b.x + w;
        y0 = b.y;
        h = b.h;
        y1 = b.y + h;
        objClass = b.cl;
        det_class = coco_class_name[objClass];
        cv::rectangle(imageOrig, cv::Point(x0, y0), cv::Point(x1, y1), colors[objClass], 2);
        // draw label
        cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::rectangle(imageOrig, cv::Point(x0, y0), cv::Point((x0 + textSize.width - 2), (y0 - textSize.height - 2)), colors[b.cl], -1);
        cv::putText(imageOrig, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
        
    }
    cv::namedWindow("cnet", cv::WINDOW_NORMAL);
    cv::imshow("cnet", imageOrig);
    cv::waitKey(10000);
    TIMER_STOP
    return 0;
}
