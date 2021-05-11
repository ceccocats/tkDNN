#include <iostream>
#include "tkdnn.h"
const char *input_bin = "dla34_ctrack/debug/input_base-level0-0.bin";
// const char *input_bin = "dla34_ctrack/debug/input.bin";
// const char *pre_img_bin = "dla34_ctrack/debug/pre_imgages.bin";
// const char *pre_hm_bin = "dla34_ctrack/debug/pre_hms.bin";
// //pre
// const char *pre_img_conv1_bin = "dla34_ctrack/layers/base-pre_img_layer-0.bin";
// const char *pre_hm_conv1_bin = "dla34_ctrack/layers/base-pre_hm_layer-0.bin";
// const char *conv1_bin = "dla34_ctrack/layers/base-base_layer-0.bin";

const char *conv2_bin = "dla34_ctrack/layers/base-level0-0.bin";
const char *conv3_bin = "dla34_ctrack/layers/base-level1-0.bin";
// s - stage, t - tree
const char *s1_t1_conv1_bin = "dla34_ctrack/layers/base-level2-tree1-conv1.bin";
const char *s1_t1_conv2_bin = "dla34_ctrack/layers/base-level2-tree1-conv2.bin";
const char *s1_t1_project = "dla34_ctrack/layers/base-level2-project-0.bin";
const char *s1_t2_conv1_bin = "dla34_ctrack/layers/base-level2-tree2-conv1.bin";
const char *s1_t2_conv2_bin = "dla34_ctrack/layers/base-level2-tree2-conv2.bin";
const char *s1_root_conv1_bin = "dla34_ctrack/layers/base-level2-root-conv.bin";
const char *s2_t1_t1_conv1_bin = "dla34_ctrack/layers/base-level3-tree1-tree1-conv1.bin";
const char *s2_t1_t1_conv2_bin = "dla34_ctrack/layers/base-level3-tree1-tree1-conv2.bin";
const char *s2_t1_t1_project = "dla34_ctrack/layers/base-level3-tree1-project-0.bin";
const char *s2_t1_t2_conv1_bin = "dla34_ctrack/layers/base-level3-tree1-tree2-conv1.bin";
const char *s2_t1_t2_conv2_bin = "dla34_ctrack/layers/base-level3-tree1-tree2-conv2.bin";
const char *s2_t1_root_conv1_bin = "dla34_ctrack/layers/base-level3-tree1-root-conv.bin";
const char *s2_t2_t1_conv1_bin = "dla34_ctrack/layers/base-level3-tree2-tree1-conv1.bin";
const char *s2_t2_t1_conv2_bin = "dla34_ctrack/layers/base-level3-tree2-tree1-conv2.bin";
const char *s2_t2_t2_conv1_bin = "dla34_ctrack/layers/base-level3-tree2-tree2-conv1.bin";
const char *s2_t2_t2_conv2_bin = "dla34_ctrack/layers/base-level3-tree2-tree2-conv2.bin";
const char *s2_t2_root_conv1_bin = "dla34_ctrack/layers/base-level3-tree2-root-conv.bin";
const char *s3_t1_t1_conv1_bin = "dla34_ctrack/layers/base-level4-tree1-tree1-conv1.bin";
const char *s3_t1_t1_conv2_bin = "dla34_ctrack/layers/base-level4-tree1-tree1-conv2.bin";
const char *s3_t1_t1_project = "dla34_ctrack/layers/base-level4-tree1-project-0.bin";
const char *s3_t1_t2_conv1_bin = "dla34_ctrack/layers/base-level4-tree1-tree2-conv1.bin";
const char *s3_t1_t2_conv2_bin = "dla34_ctrack/layers/base-level4-tree1-tree2-conv2.bin";
const char *s3_t1_root_conv1_bin = "dla34_ctrack/layers/base-level4-tree1-root-conv.bin";
const char *s3_t2_t1_conv1_bin = "dla34_ctrack/layers/base-level4-tree2-tree1-conv1.bin";
const char *s3_t2_t1_conv2_bin = "dla34_ctrack/layers/base-level4-tree2-tree1-conv2.bin";
const char *s3_t2_t2_conv1_bin = "dla34_ctrack/layers/base-level4-tree2-tree2-conv1.bin";
const char *s3_t2_t2_conv2_bin = "dla34_ctrack/layers/base-level4-tree2-tree2-conv2.bin";
const char *s3_t2_root_conv1_bin = "dla34_ctrack/layers/base-level4-tree2-root-conv.bin";
const char *s4_t1_conv1_bin = "dla34_ctrack/layers/base-level5-tree1-conv1.bin";
const char *s4_t1_conv2_bin = "dla34_ctrack/layers/base-level5-tree1-conv2.bin";
const char *s4_t1_project = "dla34_ctrack/layers/base-level5-project-0.bin";
const char *s4_t2_conv1_bin = "dla34_ctrack/layers/base-level5-tree2-conv1.bin";
const char *s4_t2_conv2_bin = "dla34_ctrack/layers/base-level5-tree2-conv2.bin";
const char *s4_root_conv1_bin = "dla34_ctrack/layers/base-level5-root-conv.bin";

//final
// const char *fc_bin = "dla34_ctrack/layers/output.bin";

const char *ida_0_p_1_dcn_bin = "dla34_ctrack/layers/dla_up-ida_0-proj_1-conv.bin";
const char *ida_0_p_1_conv_bin = "dla34_ctrack/layers/dla_up-ida_0-proj_1-conv-conv_offset_mask.bin";
const char *ida_0_up_1_deconv_bin = "dla34_ctrack/layers/dla_up-ida_0-up_1.bin";
const char *ida_0_n_1_dcn_bin = "dla34_ctrack/layers/dla_up-ida_0-node_1-conv.bin";
const char *ida_0_n_1_conv_bin = "dla34_ctrack/layers/dla_up-ida_0-node_1-conv-conv_offset_mask.bin";

const char *ida_1_p_1_dcn_bin = "dla34_ctrack/layers/dla_up-ida_1-proj_1-conv.bin";
const char *ida_1_p_1_conv_bin = "dla34_ctrack/layers/dla_up-ida_1-proj_1-conv-conv_offset_mask.bin";
const char *ida_1_up_1_deconv_bin = "dla34_ctrack/layers/dla_up-ida_1-up_1.bin";
const char *ida_1_n_1_dcn_bin = "dla34_ctrack/layers/dla_up-ida_1-node_1-conv.bin";
const char *ida_1_n_1_conv_bin = "dla34_ctrack/layers/dla_up-ida_1-node_1-conv-conv_offset_mask.bin";
const char *ida_1_p_2_dcn_bin = "dla34_ctrack/layers/dla_up-ida_1-proj_2-conv.bin";
const char *ida_1_p_2_conv_bin = "dla34_ctrack/layers/dla_up-ida_1-proj_2-conv-conv_offset_mask.bin";
const char *ida_1_up_2_deconv_bin = "dla34_ctrack/layers/dla_up-ida_1-up_2.bin";
const char *ida_1_n_2_dcn_bin = "dla34_ctrack/layers/dla_up-ida_1-node_2-conv.bin";
const char *ida_1_n_2_conv_bin = "dla34_ctrack/layers/dla_up-ida_1-node_2-conv-conv_offset_mask.bin";

const char *ida_2_p_1_dcn_bin = "dla34_ctrack/layers/dla_up-ida_2-proj_1-conv.bin";
const char *ida_2_p_1_conv_bin = "dla34_ctrack/layers/dla_up-ida_2-proj_1-conv-conv_offset_mask.bin";
const char *ida_2_up_1_deconv_bin = "dla34_ctrack/layers/dla_up-ida_2-up_1.bin";
const char *ida_2_n_1_dcn_bin = "dla34_ctrack/layers/dla_up-ida_2-node_1-conv.bin";
const char *ida_2_n_1_conv_bin = "dla34_ctrack/layers/dla_up-ida_2-node_1-conv-conv_offset_mask.bin";
const char *ida_2_p_2_dcn_bin = "dla34_ctrack/layers/dla_up-ida_2-proj_2-conv.bin";
const char *ida_2_p_2_conv_bin = "dla34_ctrack/layers/dla_up-ida_2-proj_2-conv-conv_offset_mask.bin";
const char *ida_2_up_2_deconv_bin = "dla34_ctrack/layers/dla_up-ida_2-up_2.bin";
const char *ida_2_n_2_dcn_bin = "dla34_ctrack/layers/dla_up-ida_2-node_2-conv.bin";
const char *ida_2_n_2_conv_bin = "dla34_ctrack/layers/dla_up-ida_2-node_2-conv-conv_offset_mask.bin";
const char *ida_2_p_3_dcn_bin = "dla34_ctrack/layers/dla_up-ida_2-proj_3-conv.bin";
const char *ida_2_p_3_conv_bin = "dla34_ctrack/layers/dla_up-ida_2-proj_3-conv-conv_offset_mask.bin";
const char *ida_2_up_3_deconv_bin = "dla34_ctrack/layers/dla_up-ida_2-up_3.bin";
const char *ida_2_n_3_dcn_bin = "dla34_ctrack/layers/dla_up-ida_2-node_3-conv.bin";
const char *ida_2_n_3_conv_bin = "dla34_ctrack/layers/dla_up-ida_2-node_3-conv-conv_offset_mask.bin";

const char *ida_up_p_1_dcn_bin = "dla34_ctrack/layers/ida_up-proj_1-conv.bin";
const char *ida_up_p_1_conv_bin = "dla34_ctrack/layers/ida_up-proj_1-conv-conv_offset_mask.bin";
const char *ida_up_up_1_deconv_bin = "dla34_ctrack/layers/ida_up-up_1.bin";
const char *ida_up_n_1_dcn_bin = "dla34_ctrack/layers/ida_up-node_1-conv.bin";
const char *ida_up_n_1_conv_bin = "dla34_ctrack/layers/ida_up-node_1-conv-conv_offset_mask.bin";
const char *ida_up_p_2_dcn_bin = "dla34_ctrack/layers/ida_up-proj_2-conv.bin";
const char *ida_up_p_2_conv_bin = "dla34_ctrack/layers/ida_up-proj_2-conv-conv_offset_mask.bin";
const char *ida_up_up_2_deconv_bin = "dla34_ctrack/layers/ida_up-up_2.bin";
const char *ida_up_n_2_dcn_bin = "dla34_ctrack/layers/ida_up-node_2-conv.bin";
const char *ida_up_n_2_conv_bin = "dla34_ctrack/layers/ida_up-node_2-conv-conv_offset_mask.bin";

const char *hm_conv1_bin = "dla34_ctrack/layers/hm-0.bin";
const char *hm_conv2_bin = "dla34_ctrack/layers/hm-2.bin";
const char *wh_conv1_bin = "dla34_ctrack/layers/wh-0.bin";
const char *wh_conv2_bin = "dla34_ctrack/layers/wh-2.bin";
const char *reg_conv1_bin = "dla34_ctrack/layers/reg-0.bin";
const char *reg_conv2_bin = "dla34_ctrack/layers/reg-2.bin";
const char *track_conv1_bin = "dla34_ctrack/layers/tracking-0.bin";
const char *track_conv2_bin = "dla34_ctrack/layers/tracking-2.bin";
const char *dep_conv1_bin = "dla34_ctrack/layers/dep-0.bin";
const char *dep_conv2_bin = "dla34_ctrack/layers/dep-2.bin";
const char *rot_conv1_bin = "dla34_ctrack/layers/rot-0.bin";
const char *rot_conv2_bin = "dla34_ctrack/layers/rot-2.bin";
const char *dim_conv1_bin = "dla34_ctrack/layers/dim-0.bin";
const char *dim_conv2_bin = "dla34_ctrack/layers/dim-2.bin";
const char *a_off_conv1_bin = "dla34_ctrack/layers/amodel_offset-0.bin";
const char *a_off_conv2_bin = "dla34_ctrack/layers/amodel_offset-2.bin";

const char *output_bin[]={
"dla34_ctrack/debug/hm.bin",
"dla34_ctrack/debug/wh.bin",
"dla34_ctrack/debug/reg.bin",
"dla34_ctrack/debug/tracking.bin",
"dla34_ctrack/debug/dep.bin",
"dla34_ctrack/debug/rot.bin",
"dla34_ctrack/debug/dim.bin",
"dla34_ctrack/debug/amodel_offset.bin"};
// const char *output_bin = "dla34_ctrack/debug/base-level0-2.bin";
int main()
{

    downloadWeightsifDoNotExist("dla34_ctrack/debug/input.bin", "dla34_ctrack", "https://cloud.hipert.unimore.it/s/rjNfgGL9FtAXLHp/download");

    // Network layout
    // tk::dnn::dataDim_t dim_in0(1, 3, 512, 512, 1);
    // tk::dnn::dataDim_t dim_in1(1, 1, 512, 512, 1);
    tk::dnn::dataDim_t dim_in0(1, 16, 512, 512, 1);
    // dnnType *i0_d, *i1_d, *i2_d;
    // dnnType *i0_h, *i1_h, *i2_h;
    // checkCuda( cudaMalloc(&i0_d, dim_in0.tot()*sizeof(dnnType)) );
    // checkCuda( cudaMalloc(&i1_d, dim_in1.tot()*sizeof(dnnType)) );
    // checkCuda( cudaMalloc(&i2_d, dim_in0.tot()*sizeof(dnnType)) );

    tk::dnn::Network net(dim_in0);
    tk::dnn::Layer *last1, *last2, *last3, *last4;
    tk::dnn::Layer *base1, *base2, *base3, *base4, *base5, *base6, *ida1, *ida2_1, *ida2_2, *ida3_1, *ida3_2, *ida3_3, *idaup_1, *idaup_2;
    // tk::dnn::Layer *pre_img, *pre_hm;

    // //pre-img
    // readBinaryFile(pre_img_bin, dim_in0.tot(), &i0_h, &i0_d);
    // tk::dnn::Input *in_pre_img = new tk::dnn::Input(&net, dim_in0, i0_d);
    // tk::dnn::Conv2d     pre_img_conv1(&net, 16, 7, 7, 1, 1, 3, 3, pre_img_conv1_bin, true);
    // tk::dnn::Activation pre_img_relu(&net, CUDNN_ACTIVATION_RELU);
    // pre_img = &pre_img_relu;

    // //pre-hm
    // readBinaryFile(pre_hm_bin, dim_in1.tot(), &i1_h, &i1_d);
    // tk::dnn::Input *in_pre_hm = new tk::dnn::Input(&net, dim_in1, i1_d);
    // tk::dnn::Conv2d     pre_hm_conv1(&net, 16, 7, 7, 1, 1, 3, 3, pre_hm_conv1_bin, true);
    // tk::dnn::Activation pre_hm_relu(&net, CUDNN_ACTIVATION_RELU);
    // pre_hm = &pre_hm_relu;

    // // image input
    // readBinaryFile(input_bin, dim_in0.tot(), &i2_h, &i2_d);
    // tk::dnn::Input *input_image = new tk::dnn::Input(&net, dim_in0, i2_d);
    // tk::dnn::Conv2d *conv1 = new tk::dnn::Conv2d(&net, 16, 7, 7, 1, 1, 3, 3, conv1_bin, true);
    // tk::dnn::Activation relu1(&net, CUDNN_ACTIVATION_RELU);

    // tk::dnn::Shortcut s0_input(&net, pre_img);
    // tk::dnn::Shortcut s1_input(&net, pre_hm);
    
    tk::dnn::Conv2d conv2(&net, 16, 3, 3, 1, 1, 1, 1, conv2_bin, true);
    tk::dnn::Activation relu2(&net, CUDNN_ACTIVATION_RELU);
    base1 = &relu2;
  
    tk::dnn::Conv2d conv3(&net, 32, 3, 3, 2, 2, 1, 1, conv3_bin, true);
    tk::dnn::Activation relu3(&net, CUDNN_ACTIVATION_RELU);
    base2 = &relu3;
    
    // level 2
    // tree 1
        tk::dnn::Conv2d     s1_t1_conv1(&net, 64, 3, 3, 2, 2, 1, 1, s1_t1_conv1_bin, true);
        tk::dnn::Activation s1_t1_relu1(&net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d     s1_t1_conv2(&net, 64, 3, 3, 1, 1, 1, 1, s1_t1_conv2_bin, true);
        last2 = &s1_t1_conv2;

        // get the basicblock input and apply maxpool conv2d and relu
        tk::dnn::Layer      *route_s1_t1_layers[1] = { base2 };
        tk::dnn::Route      route_s1_t1(&net, route_s1_t1_layers, 1);
        // downsample
        tk::dnn::Pooling    s1_t1_maxpool1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
        // project
        tk::dnn::Conv2d     s1_t1_residual1_conv1(&net, 64, 1, 1, 1, 1, 0, 0, s1_t1_project, true);
        
        tk::dnn::Shortcut   s1_t1_s1(&net, last2);
        tk::dnn::Activation s1_t1_relu(&net, CUDNN_ACTIVATION_RELU);
    
    last1 = &s1_t1_relu;
    // tree 2
        tk::dnn::Conv2d     s1_t2_conv1(&net, 64, 3, 3, 1, 1, 1, 1, s1_t2_conv1_bin, true);
        tk::dnn::Activation s1_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d     s1_t2_conv2(&net, 64, 3, 3, 1, 1, 1, 1, s1_t2_conv2_bin, true);
        
        tk::dnn::Shortcut   s1_t2_s1(&net, last1);
        tk::dnn::Activation s1_t2_relu(&net, CUDNN_ACTIVATION_RELU);
        last2 = &s1_t2_relu;

    // root
        // join last1 and net in single input 128, 56, 56
        tk::dnn::Layer      *route_s1_root_layers[2] = { last2, last1 };
        tk::dnn::Route      route_s1_root(&net, route_s1_root_layers, 2);
        tk::dnn::Conv2d     s1_root_conv1(&net, 64, 1, 1, 1, 1, 0, 0, s1_root_conv1_bin, true);
        tk::dnn::Activation s1_root_relu(&net, CUDNN_ACTIVATION_RELU);

    base3 = &s1_root_relu;

    // level 3
    // tree 1
        // tree 1
            tk::dnn::Conv2d     s2_t1_t1_conv1(&net, 128, 3, 3, 2, 2, 1, 1, s2_t1_t1_conv1_bin, true);
            tk::dnn::Activation s2_t1_t1_relu1(&net, CUDNN_ACTIVATION_RELU);      
        
            tk::dnn::Conv2d     s2_t1_t1_conv2(&net, 128, 3, 3, 1, 1, 1, 1, s2_t1_t1_conv2_bin, true);
            last2 = &s2_t1_t1_conv2;

            // get the basicblock input and apply maxpool conv2d and relu
            tk::dnn::Layer      *route_s2_t1_t1_layers[1] = { base3 };
            tk::dnn::Route      route_s2_t1_t1(&net, route_s2_t1_t1_layers, 1);
            // downsample
            tk::dnn::Pooling    s2_t1_t1_maxpool1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
            last4 = &s2_t1_t1_maxpool1;
            // project
            tk::dnn::Conv2d     s2_t1_t1_residual1_conv1(&net, 128, 1, 1, 1, 1, 0, 0, s2_t1_t1_project, true);
            
            tk::dnn::Shortcut   s2_t1_t1_s1(&net, last2);
            tk::dnn::Activation s2_t1_t1_relu(&net, CUDNN_ACTIVATION_RELU);

        last1 = &s2_t1_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d     s2_t1_t2_conv1(&net, 128, 3, 3, 1, 1, 1, 1, s2_t1_t2_conv1_bin, true);
            tk::dnn::Activation s2_t1_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     s2_t1_t2_conv2(&net, 128, 3, 3, 1, 1, 1, 1, s2_t1_t2_conv2_bin, true);
            
            tk::dnn::Shortcut   s2_t1_t2_s1(&net, last1);
            tk::dnn::Activation s2_t1_t2_relu(&net, CUDNN_ACTIVATION_RELU);
            last2 = &s2_t1_t2_relu;

        // root
            // join last1 and net in single input 128, 56, 56
            tk::dnn::Layer      *route_s2_t1_root_layers[2] = { last2, last1 };
            tk::dnn::Route      route_s2_t1_root(&net, route_s2_t1_root_layers, 2);
            tk::dnn::Conv2d     s2_t1_root_conv1(&net, 128, 1, 1, 1, 1, 0, 0, s2_t1_root_conv1_bin, true);
            tk::dnn::Activation s2_t1_root_relu(&net, CUDNN_ACTIVATION_RELU);
    
    last1 = &s2_t1_root_relu;
    last3 = &s2_t1_root_relu;
    // tree 2
        // tree 1
            tk::dnn::Conv2d     s2_t2_t1_conv1(&net, 128, 3, 3, 1, 1, 1, 1, s2_t2_t1_conv1_bin, true);
            tk::dnn::Activation s2_t2_t1_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     s2_t2_t1_conv2(&net, 128, 3, 3, 1, 1, 1, 1, s2_t2_t1_conv2_bin, true);
            tk::dnn::Shortcut   s2_t2_t1_s1(&net, last1);
            tk::dnn::Activation s2_t2_t1_relu(&net, CUDNN_ACTIVATION_RELU);

        last1 = &s2_t2_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d     s2_t2_t2_conv1(&net, 128, 3, 3, 1, 1, 1, 1, s2_t2_t2_conv1_bin, true);
            tk::dnn::Activation s2_t2_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     s2_t2_t2_conv2(&net, 128, 3, 3, 1, 1, 1, 1, s2_t2_t2_conv2_bin, true);
            
            tk::dnn::Shortcut   s2_t2_t2_s1(&net, last1);
            tk::dnn::Activation s2_t2_t2_relu(&net, CUDNN_ACTIVATION_RELU);
            last2 = &s2_t2_t2_relu;

        // root
            // join last1 and net in single input 128, 56, 56
            tk::dnn::Layer      *route_s2_t2_root_layers[4] = { last2, last1, last4, last3};
            tk::dnn::Route      route_s2_t2_root(&net, route_s2_t2_root_layers, 4);
            tk::dnn::Conv2d     s2_t2_root_conv1(&net, 128, 1, 1, 1, 1, 0, 0, s2_t2_root_conv1_bin, true);
            tk::dnn::Activation s2_t2_root_relu(&net, CUDNN_ACTIVATION_RELU);
    
    base4 = &s2_t2_root_relu;

    // level 4
    // tree 1
        // tree 1
            tk::dnn::Conv2d     s3_t1_t1_conv1(&net, 256, 3, 3, 2, 2, 1, 1, s3_t1_t1_conv1_bin, true);
            tk::dnn::Activation s3_t1_t1_relu1(&net, CUDNN_ACTIVATION_RELU);      
        
            tk::dnn::Conv2d     s3_t1_t1_conv2(&net, 256, 3, 3, 1, 1, 1, 1, s3_t1_t1_conv2_bin, true);
            last2 = &s3_t1_t1_conv2;

            // get the basicblock input and apply maxpool conv2d and relu
            tk::dnn::Layer      *route_s3_t1_t1_layers[1] = { base4 };
            tk::dnn::Route      route_s3_t1_t1(&net, route_s3_t1_t1_layers, 1);
            // downsample
            tk::dnn::Pooling    s3_t1_t1_maxpool1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
            last4 = &s3_t1_t1_maxpool1;
            // project
            tk::dnn::Conv2d     s3_t1_t1_residual1_conv1(&net, 256, 1, 1, 1, 1, 0, 0, s3_t1_t1_project, true);
            
            tk::dnn::Shortcut   s3_t1_t1_s1(&net, last2);
            tk::dnn::Activation s3_t1_t1_relu(&net, CUDNN_ACTIVATION_RELU);

        last1 = &s3_t1_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d     s3_t1_t2_conv1(&net, 256, 3, 3, 1, 1, 1, 1, s3_t1_t2_conv1_bin, true);
            tk::dnn::Activation s3_t1_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     s3_t1_t2_conv2(&net, 256, 3, 3, 1, 1, 1, 1, s3_t1_t2_conv2_bin, true);
            
            tk::dnn::Shortcut   s3_t1_t2_s1(&net, last1);
            tk::dnn::Activation s3_t1_t2_relu(&net, CUDNN_ACTIVATION_RELU);
            last2 = &s3_t1_t2_relu;

        // root
            // join last1 and net in single input 256, 56, 56
            tk::dnn::Layer      *route_s3_t1_root_layers[2] = { last2, last1 };
            tk::dnn::Route      route_s3_t1_root(&net, route_s3_t1_root_layers, 2);
            tk::dnn::Conv2d     s3_t1_root_conv1(&net, 256, 1, 1, 1, 1, 0, 0, s3_t1_root_conv1_bin, true);
            tk::dnn::Activation s3_t1_root_relu(&net, CUDNN_ACTIVATION_RELU);
    
    last1 = &s3_t1_root_relu;
    last3 = &s3_t1_root_relu;
    // tree 2
        // tree 1
            tk::dnn::Conv2d     s3_t2_t1_conv1(&net, 256, 3, 3, 1, 1, 1, 1, s3_t2_t1_conv1_bin, true);
            tk::dnn::Activation s3_t2_t1_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     s3_t2_t1_conv2(&net, 256, 3, 3, 1, 1, 1, 1, s3_t2_t1_conv2_bin, true);
            tk::dnn::Shortcut   s3_t2_t1_s1(&net, last1);
            tk::dnn::Activation s3_t2_t1_relu(&net, CUDNN_ACTIVATION_RELU);

        last1 = &s3_t2_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d     s3_t2_t2_conv1(&net, 256, 3, 3, 1, 1, 1, 1, s3_t2_t2_conv1_bin, true);
            tk::dnn::Activation s3_t2_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d     s3_t2_t2_conv2(&net, 256, 3, 3, 1, 1, 1, 1, s3_t2_t2_conv2_bin, true);
            
            tk::dnn::Shortcut   s3_t2_t2_s1(&net, last1);
            tk::dnn::Activation s3_t2_t2_relu(&net, CUDNN_ACTIVATION_RELU);
            last2 = &s3_t2_t2_relu;

        // root
            // join last1 and net in single input 256, 56, 56
            tk::dnn::Layer      *route_s3_t2_root_layers[4] = { last2, last1, last4, last3};
            tk::dnn::Route      route_s3_t2_root(&net, route_s3_t2_root_layers, 4);
            tk::dnn::Conv2d     s3_t2_root_conv1(&net, 256, 1, 1, 1, 1, 0, 0, s3_t2_root_conv1_bin, true);
            tk::dnn::Activation s3_t2_root_relu(&net, CUDNN_ACTIVATION_RELU);
    
    base5 = &s3_t2_root_relu;

    // level 5
    // tree 1
        tk::dnn::Conv2d     s4_t1_conv1(&net, 512, 3, 3, 2, 2, 1, 1, s4_t1_conv1_bin, true);
        tk::dnn::Activation s4_t1_relu1(&net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d     s4_t1_conv2(&net, 512, 3, 3, 1, 1, 1, 1, s4_t1_conv2_bin, true);
        last2 = &s4_t1_conv2;

        // get the basicblock input and apply maxpool conv2d and relu
        tk::dnn::Layer      *route_s4_t1_layers[1] = { base5 };
        tk::dnn::Route      route_s4_t1(&net, route_s4_t1_layers, 1);
        // downsample
        tk::dnn::Pooling    s4_t1_maxpool1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
        last4 = &s4_t1_maxpool1;
        // project
        tk::dnn::Conv2d     s4_t1_residual1_conv1(&net, 512, 1, 1, 1, 1, 0, 0, s4_t1_project, true);
        
        tk::dnn::Shortcut   s4_t1_s1(&net, last2);
        tk::dnn::Activation s4_t1_relu(&net, CUDNN_ACTIVATION_RELU);

    last1 = &s4_t1_relu;
    
    // tree 2
        tk::dnn::Conv2d     s4_t2_conv1(&net, 512, 3, 3, 1, 1, 1, 1, s4_t2_conv1_bin, true);
        tk::dnn::Activation s4_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d     s4_t2_conv2(&net, 512, 3, 3, 1, 1, 1, 1, s4_t2_conv2_bin, true);
        
        tk::dnn::Shortcut   s4_t2_s1(&net, last1);
        tk::dnn::Activation s4_t2_relu(&net, CUDNN_ACTIVATION_RELU);
        last2 = &s4_t2_relu;

    // root
        // join last1 and net in single input 128, 56, 56
        tk::dnn::Layer      *route_s4_root_layers[3] = { last2, last1, last4 };
        tk::dnn::Route      route_s4_root(&net, route_s4_root_layers, 3);
        tk::dnn::Conv2d     s4_root_conv1(&net, 512, 1, 1, 1, 1, 0, 0, s4_root_conv1_bin, true);
        tk::dnn::Activation s4_root_relu(&net, CUDNN_ACTIVATION_RELU);

    base6 = &s4_root_relu;
 
    //final
    // tk::dnn::Pooling avgpool(&net, 7, 7, 7, 7, 0, 0, tk::dnn::POOLING_AVERAGE);
    // tk::dnn::Dense   fc(&net, 1000, fc_bin);
    
    //ida 0 
    tk::dnn::Layer          *route_ida0[1] = { base6 };
    tk::dnn::Route          route_ida0_0(&net, route_ida0, 1);

    tk::dnn::DeformConv2d   ida_0_p_1_dcn(&net, 256, 1, 3, 3, 1, 1, 1, 1, ida_0_p_1_dcn_bin, ida_0_p_1_conv_bin, true);
    tk::dnn::Activation     ida_0_p_1_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       ida_0_up_1_deconv(&net, 256, 4, 4, 2, 2, 1, 1, ida_0_up_1_deconv_bin, false, 256);
    tk::dnn::Shortcut       ida_0_shortcut(&net, base5);    
    tk::dnn::DeformConv2d   ida_0_n_1_dcn(&net, 256, 1, 3, 3, 1, 1, 1, 1, ida_0_n_1_dcn_bin, ida_0_n_1_conv_bin, true);
    tk::dnn::Activation     ida_0_n_1_relu(&net, CUDNN_ACTIVATION_RELU);
    ida1 = &ida_0_n_1_relu;
  
    //ida1-1
    tk::dnn::Layer          *route_ida1_layers_1[1] = { base5 };
    tk::dnn::Route          route_ida1_1(&net, route_ida1_layers_1, 1);
    
    tk::dnn::DeformConv2d   ida_1_p_1_dcn(&net, 128, 1, 3, 3, 1, 1, 1, 1, ida_1_p_1_dcn_bin, ida_1_p_1_conv_bin, true);
    tk::dnn::Activation     ida_1_p_1_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       ida_1_up_1_deconv(&net, 128, 4, 4, 2, 2, 1, 1, ida_1_up_1_deconv_bin, false, 128);
    tk::dnn::Shortcut       ida_1_shortcut1(&net, base4);    
    tk::dnn::DeformConv2d   ida_1_n_1_dcn(&net, 128, 1, 3, 3, 1, 1, 1, 1, ida_1_n_1_dcn_bin, ida_1_n_1_conv_bin, true);
    tk::dnn::Activation     ida_1_n_1_relu(&net, CUDNN_ACTIVATION_RELU);
    ida2_1 = &ida_1_n_1_relu;

    //ida1-2
    tk::dnn::Layer          *route_ida1_layers_2[1] = { ida1 };
    tk::dnn::Route          route_ida1_2(&net, route_ida1_layers_2, 1);

    tk::dnn::DeformConv2d   ida_1_p_2_dcn(&net, 128, 1, 3, 3, 1, 1, 1, 1, ida_1_p_2_dcn_bin, ida_1_p_2_conv_bin, true);
    tk::dnn::Activation     ida_1_p_2_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       ida_1_up_2_deconv(&net, 128, 4, 4, 2, 2, 1, 1, ida_1_up_2_deconv_bin, false, 128);
    tk::dnn::Shortcut       ida_1_shortcut2(&net, ida2_1);    
    tk::dnn::DeformConv2d   ida_1_n_2_dcn(&net, 128, 1, 3, 3, 1, 1, 1, 1, ida_1_n_2_dcn_bin, ida_1_n_2_conv_bin, true);
    tk::dnn::Activation     ida_1_n_2_relu(&net, CUDNN_ACTIVATION_RELU);
    ida2_2 = &ida_1_n_2_relu;

    //ida2-1
    tk::dnn::Layer          *route_ida2_layers_1[1] = { base4 };
    tk::dnn::Route          route_ida2_1(&net, route_ida2_layers_1, 1);
    
    tk::dnn::DeformConv2d   ida_2_p_1_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_2_p_1_dcn_bin, ida_2_p_1_conv_bin, true);
    tk::dnn::Activation     ida_2_p_1_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       ida_2_up_1_deconv(&net, 64, 4, 4, 2, 2, 1, 1, ida_2_up_1_deconv_bin, false, 64);
    tk::dnn::Shortcut       ida_2_shortcut1(&net, base3);    
    tk::dnn::DeformConv2d   ida_2_n_1_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_2_n_1_dcn_bin, ida_2_n_1_conv_bin, true);
    tk::dnn::Activation     ida_2_n_1_relu(&net, CUDNN_ACTIVATION_RELU);
    ida3_1 = &ida_2_n_1_relu;

    //ida2-2
    tk::dnn::Layer          *route_ida2_layers_2[1] = { ida2_1 };
    tk::dnn::Route          route_ida2_2(&net, route_ida2_layers_2, 1);
    
    tk::dnn::DeformConv2d   ida_2_p_2_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_2_p_2_dcn_bin, ida_2_p_2_conv_bin, true);
    tk::dnn::Activation     ida_2_p_2_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       ida_2_up_2_deconv(&net, 64, 4, 4, 2, 2, 1, 1, ida_2_up_2_deconv_bin, false, 64);
    tk::dnn::Shortcut       ida_2_shortcut2(&net, ida3_1);    
    tk::dnn::DeformConv2d   ida_2_n_2_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_2_n_2_dcn_bin, ida_2_n_2_conv_bin, true);
    tk::dnn::Activation     ida_2_n_2_relu(&net, CUDNN_ACTIVATION_RELU);
    ida3_2 = &ida_2_n_2_relu;

    //ida2-3
    tk::dnn::Layer          *route_ida2_layers_3[1] = { ida2_2 };
    tk::dnn::Route          route_ida2_3(&net, route_ida2_layers_3, 1);
    
    tk::dnn::DeformConv2d   ida_2_p_3_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_2_p_3_dcn_bin, ida_2_p_3_conv_bin, true);
    tk::dnn::Activation     ida_2_p_3_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       ida_2_up_3_deconv(&net, 64, 4, 4, 2, 2, 1, 1, ida_2_up_3_deconv_bin, false, 64);
    tk::dnn::Shortcut       ida_2_shortcut3(&net, ida3_2);    
    tk::dnn::DeformConv2d   ida_2_n_3_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_2_n_3_dcn_bin, ida_2_n_3_conv_bin, true);
    tk::dnn::Activation     ida_2_n_3_relu(&net, CUDNN_ACTIVATION_RELU);
    ida3_3 = &ida_2_n_3_relu;

    //idaup-1
    tk::dnn::Layer          *route_idaup_layers_1[1] = { ida2_2 };
    tk::dnn::Route          route_idaup_1(&net, route_idaup_layers_1, 1);
    
    tk::dnn::DeformConv2d   idaup_p_1_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_up_p_1_dcn_bin, ida_up_p_1_conv_bin, true);
    tk::dnn::Activation     idaup_p_1_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       idaup_up_1_deconv(&net, 64, 4, 4, 2, 2, 1, 1, ida_up_up_1_deconv_bin, false, 64);
    tk::dnn::Shortcut       idaup_shortcut1(&net, ida3_3);    
    tk::dnn::DeformConv2d   idaup_n_1_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_up_n_1_dcn_bin, ida_up_n_1_conv_bin, true);
    tk::dnn::Activation     idaup_n_1_relu(&net, CUDNN_ACTIVATION_RELU);
    idaup_1 = &idaup_n_1_relu;

    //idaup-2
    tk::dnn::Layer          *route_idaup_layers_2[1] = { ida1 };
    tk::dnn::Route          route_idaup_2(&net, route_idaup_layers_2, 1);

    tk::dnn::DeformConv2d   idaup_p_2_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_up_p_2_dcn_bin, ida_up_p_2_conv_bin, true);
    tk::dnn::Activation     idaup_p_2_relu(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::DeConv2d       idaup_up_2_deconv(&net, 64, 8, 8, 4, 4, 2, 2, ida_up_up_2_deconv_bin, false, 64);
    tk::dnn::Shortcut       idaup_shortcut2(&net, idaup_1);    
    tk::dnn::DeformConv2d   idaup_n_2_dcn(&net, 64, 1, 3, 3, 1, 1, 1, 1, ida_up_n_2_dcn_bin, ida_up_n_2_conv_bin, true);
    tk::dnn::Activation     idaup_n_2_relu(&net, CUDNN_ACTIVATION_RELU);
    idaup_2 = &idaup_n_2_relu;
 
    tk::dnn::Layer    *route_1_0_layers[1] = { idaup_2 };

    // hm
    tk::dnn::Conv2d     *hm_conv1 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, hm_conv1_bin, false);
    tk::dnn::Activation *hm_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *hm = new tk::dnn::Conv2d(&net, 10, 1, 1, 1, 1, 0, 0, hm_conv2_bin, false);
    hm->setFinal();

    int kernel = 3; 
    int pad = (kernel - 1)/2;
    tk::dnn::Activation *hm_sig      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_SIGMOID);
    tk::dnn::Pooling  *hmax                 = new tk::dnn::Pooling(&net, kernel, kernel, 1, 1, pad, pad, tk::dnn::POOLING_MAX);
    hmax->setFinal();

    // wh
    tk::dnn::Route    *route_1_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *wh_conv1 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, wh_conv1_bin, false);
    tk::dnn::Activation *wh_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *wh = new tk::dnn::Conv2d(&net, 2, 1, 1, 1, 1, 0, 0, wh_conv2_bin, false);        
    wh->setFinal();
   
    // reg
    tk::dnn::Route    *route_2_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *reg_conv1 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, reg_conv1_bin, false);
    tk::dnn::Activation *reg_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *reg = new tk::dnn::Conv2d(&net, 2, 1, 1, 1, 1, 0, 0, reg_conv2_bin, false);
    reg->setFinal();

    // tracking
    tk::dnn::Route    *route_3_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *track_conv1 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, track_conv1_bin, false);
    tk::dnn::Activation *track_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *track = new tk::dnn::Conv2d(&net, 2, 1, 1, 1, 1, 0, 0, track_conv2_bin, false);
    track->setFinal();

    // dep
    tk::dnn::Route    *route_4_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *dep_conv1 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, dep_conv1_bin, false);
    tk::dnn::Activation *dep_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *dep = new tk::dnn::Conv2d(&net, 1, 1, 1, 1, 1, 0, 0, dep_conv2_bin, false);
    dep->setFinal();

    // rot
    tk::dnn::Route    *route_5_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *rot_conv1 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, rot_conv1_bin, false);
    tk::dnn::Activation *rot_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *rot = new tk::dnn::Conv2d(&net, 8, 1, 1, 1, 1, 0, 0, rot_conv2_bin, false);
    rot->setFinal();

    // dim
    tk::dnn::Route    *route_6_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *dim_conv1 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, dim_conv1_bin, false);
    tk::dnn::Activation *dim_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *dim_ = new tk::dnn::Conv2d(&net, 3, 1, 1, 1, 1, 0, 0, dim_conv2_bin, false);
    dim_->setFinal();

    // amodel_offset
    tk::dnn::Route    *route_7_0             = new tk::dnn::Route(&net, route_1_0_layers, 1);
    tk::dnn::Conv2d     *a_off_conv1 = new tk::dnn::Conv2d(&net, 256, 3, 3, 1, 1, 1, 1, a_off_conv1_bin, false);
    tk::dnn::Activation *a_off_relu1      = new tk::dnn::Activation(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Conv2d     *a_off = new tk::dnn::Conv2d(&net, 2, 1, 1, 1, 1, 0, 0, a_off_conv2_bin, false);
    a_off->setFinal();


    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim_in0.tot(), &input_h, &data);
    //printDeviceVector(64, data, true);

    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("dla34_ctrack"));

    tk::dnn::dataDim_t dim1 = dim_in0; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TKDNN_TSTART
        // tk::dnn::dataDim_t dim_aus;
        // net.infer(dim_aus, nullptr);
        net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }
        
    tk::dnn::dataDim_t dim2 = dim_in0;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TKDNN_TSTART
        netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }
    // dnnType *out, *out_h;
    // int odim = net.layers[net.num_layers-1]->output_dim.tot();
    // readBinaryFile(output_bin, odim, &out_h, &out);
    // dnnType *cudnn_out;
    // cudnn_out = net.layers[net.num_layers-1]->dstData;
    // std::cout<<"CUDNN vs correct"; 
    // checkResult(odim, cudnn_out, out);
    
    
    tk::dnn::Layer *outs[8] = { hm, wh, reg, track, dep, rot, dim_, a_off}; 
    int out_count = 1;
    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0; 
    for(int i=0; i<8; i++) {
        printCenteredTitle((std::string(" RESNET CHECK RESULTS ") + std::to_string(i) + " ").c_str(), '=', 30);
        
        outs[i]->output_dim.print();
        
        dnnType *out, *out_h;
        int odim = outs[i]->output_dim.tot();
        readBinaryFile(output_bin[i], odim, &out_h, &out);

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
