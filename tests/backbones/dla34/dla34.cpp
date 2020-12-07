#include <iostream>
#include "tkdnn.h"

const char *input_bin = "dla34/debug/input.bin";
const char *conv1_bin = "dla34/layers/features-init_block-conv1-conv.bin";
const char *conv2_bin = "dla34/layers/features-init_block-conv2-conv.bin";
const char *conv3_bin = "dla34/layers/features-init_block-conv3-conv.bin";
// s - stage, t - tree
const char *s1_t1_conv1_bin = "dla34/layers/features-stage1-tree1-body-conv1-conv.bin";
const char *s1_t1_conv2_bin = "dla34/layers/features-stage1-tree1-body-conv2-conv.bin";
const char *s1_t1_project = "dla34/layers/features-stage1-tree1-project_conv-conv.bin";
const char *s1_t2_conv1_bin = "dla34/layers/features-stage1-tree2-body-conv1-conv.bin";
const char *s1_t2_conv2_bin = "dla34/layers/features-stage1-tree2-body-conv2-conv.bin";
const char *s1_root_conv1_bin = "dla34/layers/features-stage1-root-conv-conv.bin";
const char *s2_t1_t1_conv1_bin = "dla34/layers/features-stage2-tree1-tree1-body-conv1-conv.bin";
const char *s2_t1_t1_conv2_bin = "dla34/layers/features-stage2-tree1-tree1-body-conv2-conv.bin";
const char *s2_t1_t1_project = "dla34/layers/features-stage2-tree1-tree1-project_conv-conv.bin";
const char *s2_t1_t2_conv1_bin = "dla34/layers/features-stage2-tree1-tree2-body-conv1-conv.bin";
const char *s2_t1_t2_conv2_bin = "dla34/layers/features-stage2-tree1-tree2-body-conv2-conv.bin";
const char *s2_t1_root_conv1_bin = "dla34/layers/features-stage2-tree1-root-conv-conv.bin";
const char *s2_t2_t1_conv1_bin = "dla34/layers/features-stage2-tree2-tree1-body-conv1-conv.bin";
const char *s2_t2_t1_conv2_bin = "dla34/layers/features-stage2-tree2-tree1-body-conv2-conv.bin";
const char *s2_t2_t2_conv1_bin = "dla34/layers/features-stage2-tree2-tree2-body-conv1-conv.bin";
const char *s2_t2_t2_conv2_bin = "dla34/layers/features-stage2-tree2-tree2-body-conv2-conv.bin";
const char *s2_t2_root_conv1_bin = "dla34/layers/features-stage2-tree2-root-conv-conv.bin";
const char *s3_t1_t1_conv1_bin = "dla34/layers/features-stage3-tree1-tree1-body-conv1-conv.bin";
const char *s3_t1_t1_conv2_bin = "dla34/layers/features-stage3-tree1-tree1-body-conv2-conv.bin";
const char *s3_t1_t1_project = "dla34/layers/features-stage3-tree1-tree1-project_conv-conv.bin";
const char *s3_t1_t2_conv1_bin = "dla34/layers/features-stage3-tree1-tree2-body-conv1-conv.bin";
const char *s3_t1_t2_conv2_bin = "dla34/layers/features-stage3-tree1-tree2-body-conv2-conv.bin";
const char *s3_t1_root_conv1_bin = "dla34/layers/features-stage3-tree1-root-conv-conv.bin";
const char *s3_t2_t1_conv1_bin = "dla34/layers/features-stage3-tree2-tree1-body-conv1-conv.bin";
const char *s3_t2_t1_conv2_bin = "dla34/layers/features-stage3-tree2-tree1-body-conv2-conv.bin";
const char *s3_t2_t2_conv1_bin = "dla34/layers/features-stage3-tree2-tree2-body-conv1-conv.bin";
const char *s3_t2_t2_conv2_bin = "dla34/layers/features-stage3-tree2-tree2-body-conv2-conv.bin";
const char *s3_t2_root_conv1_bin = "dla34/layers/features-stage3-tree2-root-conv-conv.bin";
const char *s4_t1_conv1_bin = "dla34/layers/features-stage4-tree1-body-conv1-conv.bin";
const char *s4_t1_conv2_bin = "dla34/layers/features-stage4-tree1-body-conv2-conv.bin";
const char *s4_t1_project = "dla34/layers/features-stage4-tree1-project_conv-conv.bin";
const char *s4_t2_conv1_bin = "dla34/layers/features-stage4-tree2-body-conv1-conv.bin";
const char *s4_t2_conv2_bin = "dla34/layers/features-stage4-tree2-body-conv2-conv.bin";
const char *s4_root_conv1_bin = "dla34/layers/features-stage4-root-conv-conv.bin";

//final
const char *fc_bin = "dla34/layers/output.bin";

const char *output_bin = "dla34/debug/output.bin";  

int main()
{

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 224, 224, 1);
    tk::dnn::Network net(dim);
    tk::dnn::Layer *last1, *last2, *last3, *last4;


    tk::dnn::Conv2d conv1(&net, 16, 7, 7, 1, 1, 3, 3, conv1_bin, true);
    tk::dnn::Activation relu1(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d conv2(&net, 16, 3, 3, 1, 1, 1, 1, conv2_bin, true);
    tk::dnn::Activation relu2(&net, CUDNN_ACTIVATION_RELU);

    tk::dnn::Conv2d conv3(&net, 32, 3, 3, 2, 2, 1, 1, conv3_bin, true);
    tk::dnn::Activation relu3(&net, CUDNN_ACTIVATION_RELU);

    last1 = &relu3;
    
    // level 2
    // tree 1
        tk::dnn::Conv2d s1_t1_conv1(&net, 64, 3, 3, 2, 2, 1, 1, s1_t1_conv1_bin, true);
        tk::dnn::Activation s1_t1_relu1(&net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d s1_t1_conv2(&net, 64, 3, 3, 1, 1, 1, 1, s1_t1_conv2_bin, true);
        last2 = &s1_t1_conv2;

        // get the basicblock input and apply maxpool conv2d and relu
        tk::dnn::Layer    *route_s1_t1_layers[1] = { last1 };
        tk::dnn::Route    route_s1_t1(&net, route_s1_t1_layers, 1);
        // downsample
        tk::dnn::Pooling s1_t1_maxpool1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
        // project
        tk::dnn::Conv2d s1_t1_residual1_conv1(&net, 64, 1, 1, 1, 1, 0, 0, s1_t1_project, true);
        
        tk::dnn::Shortcut s1_t1_s1(&net, last2);
        tk::dnn::Activation   s1_t1_relu(&net, CUDNN_ACTIVATION_RELU);

    last1 = &s1_t1_relu;
    
    // tree 2
        tk::dnn::Conv2d s1_t2_conv1(&net, 64, 3, 3, 1, 1, 1, 1, s1_t2_conv1_bin, true);
        tk::dnn::Activation s1_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d s1_t2_conv2(&net, 64, 3, 3, 1, 1, 1, 1, s1_t2_conv2_bin, true);
        
        tk::dnn::Shortcut s1_t2_s1(&net, last1);
        tk::dnn::Activation   s1_t2_relu(&net, CUDNN_ACTIVATION_RELU);
        last2 = &s1_t2_relu;

    // root
        // join last1 and net in single input 128, 56, 56
        tk::dnn::Layer    *route_s1_root_layers[2] = { last2, last1 };
        tk::dnn::Route    route_s1_root(&net, route_s1_root_layers, 2);
        tk::dnn::Conv2d s1_root_conv1(&net, 64, 1, 1, 1, 1, 0, 0, s1_root_conv1_bin, true);
        tk::dnn::Activation   s1_root_relu(&net, CUDNN_ACTIVATION_RELU);

    last1 = &s1_root_relu;
    // level 3
    // tree 1
        // tree 1
            tk::dnn::Conv2d s2_t1_t1_conv1(&net, 128, 3, 3, 2, 2, 1, 1, s2_t1_t1_conv1_bin, true);
            tk::dnn::Activation s2_t1_t1_relu1(&net, CUDNN_ACTIVATION_RELU);      
        
            tk::dnn::Conv2d s2_t1_t1_conv2(&net, 128, 3, 3, 1, 1, 1, 1, s2_t1_t1_conv2_bin, true);
            last2 = &s2_t1_t1_conv2;

            // get the basicblock input and apply maxpool conv2d and relu
            tk::dnn::Layer    *route_s2_t1_t1_layers[1] = { last1 };
            tk::dnn::Route    route_s2_t1_t1(&net, route_s2_t1_t1_layers, 1);
            // downsample
            tk::dnn::Pooling s2_t1_t1_maxpool1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
            last4 = &s2_t1_t1_maxpool1;
            // project
            tk::dnn::Conv2d s2_t1_t1_residual1_conv1(&net, 128, 1, 1, 1, 1, 0, 0, s2_t1_t1_project, true);
            
            tk::dnn::Shortcut s2_t1_t1_s1(&net, last2);
            tk::dnn::Activation   s2_t1_t1_relu(&net, CUDNN_ACTIVATION_RELU);

        last1 = &s2_t1_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d s2_t1_t2_conv1(&net, 128, 3, 3, 1, 1, 1, 1, s2_t1_t2_conv1_bin, true);
            tk::dnn::Activation s2_t1_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d s2_t1_t2_conv2(&net, 128, 3, 3, 1, 1, 1, 1, s2_t1_t2_conv2_bin, true);
            
            tk::dnn::Shortcut s2_t1_t2_s1(&net, last1);
            tk::dnn::Activation   s2_t1_t2_relu(&net, CUDNN_ACTIVATION_RELU);
            last2 = &s2_t1_t2_relu;

        // root
            // join last1 and net in single input 128, 56, 56
            tk::dnn::Layer    *route_s2_t1_root_layers[2] = { last2, last1 };
            tk::dnn::Route    route_s2_t1_root(&net, route_s2_t1_root_layers, 2);
            tk::dnn::Conv2d s2_t1_root_conv1(&net, 128, 1, 1, 1, 1, 0, 0, s2_t1_root_conv1_bin, true);
            tk::dnn::Activation   s2_t1_root_relu(&net, CUDNN_ACTIVATION_RELU);
    
    last1 = &s2_t1_root_relu;
    last3 = &s2_t1_root_relu;
    // tree 2
        // tree 1
            tk::dnn::Conv2d s2_t2_t1_conv1(&net, 128, 3, 3, 1, 1, 1, 1, s2_t2_t1_conv1_bin, true);
            tk::dnn::Activation s2_t2_t1_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d s2_t2_t1_conv2(&net, 128, 3, 3, 1, 1, 1, 1, s2_t2_t1_conv2_bin, true);
            tk::dnn::Shortcut s2_t2_t1_s1(&net, last1);
            tk::dnn::Activation   s2_t2_t1_relu(&net, CUDNN_ACTIVATION_RELU);

        last1 = &s2_t2_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d s2_t2_t2_conv1(&net, 128, 3, 3, 1, 1, 1, 1, s2_t2_t2_conv1_bin, true);
            tk::dnn::Activation s2_t2_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d s2_t2_t2_conv2(&net, 128, 3, 3, 1, 1, 1, 1, s2_t2_t2_conv2_bin, true);
            
            tk::dnn::Shortcut s2_t2_t2_s1(&net, last1);
            tk::dnn::Activation   s2_t2_t2_relu(&net, CUDNN_ACTIVATION_RELU);
            last2 = &s2_t2_t2_relu;

        // root
            // join last1 and net in single input 128, 56, 56
            tk::dnn::Layer    *route_s2_t2_root_layers[4] = { last2, last1, last4, last3};
            tk::dnn::Route    route_s2_t2_root(&net, route_s2_t2_root_layers, 4);
            tk::dnn::Conv2d s2_t2_root_conv1(&net, 128, 1, 1, 1, 1, 0, 0, s2_t2_root_conv1_bin, true);
            tk::dnn::Activation   s2_t2_root_relu(&net, CUDNN_ACTIVATION_RELU);
    
    
    last1 = &s2_t2_root_relu;
    // level 4
    // tree 1
        // tree 1
            tk::dnn::Conv2d s3_t1_t1_conv1(&net, 256, 3, 3, 2, 2, 1, 1, s3_t1_t1_conv1_bin, true);
            tk::dnn::Activation s3_t1_t1_relu1(&net, CUDNN_ACTIVATION_RELU);      
        
            tk::dnn::Conv2d s3_t1_t1_conv2(&net, 256, 3, 3, 1, 1, 1, 1, s3_t1_t1_conv2_bin, true);
            last2 = &s3_t1_t1_conv2;

            // get the basicblock input and apply maxpool conv2d and relu
            tk::dnn::Layer    *route_s3_t1_t1_layers[1] = { last1 };
            tk::dnn::Route    route_s3_t1_t1(&net, route_s3_t1_t1_layers, 1);
            // downsample
            tk::dnn::Pooling s3_t1_t1_maxpool1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
            last4 = &s3_t1_t1_maxpool1;
            // project
            tk::dnn::Conv2d s3_t1_t1_residual1_conv1(&net, 256, 1, 1, 1, 1, 0, 0, s3_t1_t1_project, true);
            
            tk::dnn::Shortcut s3_t1_t1_s1(&net, last2);
            tk::dnn::Activation   s3_t1_t1_relu(&net, CUDNN_ACTIVATION_RELU);

        last1 = &s3_t1_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d s3_t1_t2_conv1(&net, 256, 3, 3, 1, 1, 1, 1, s3_t1_t2_conv1_bin, true);
            tk::dnn::Activation s3_t1_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d s3_t1_t2_conv2(&net, 256, 3, 3, 1, 1, 1, 1, s3_t1_t2_conv2_bin, true);
            
            tk::dnn::Shortcut s3_t1_t2_s1(&net, last1);
            tk::dnn::Activation   s3_t1_t2_relu(&net, CUDNN_ACTIVATION_RELU);
            last2 = &s3_t1_t2_relu;

        // root
            // join last1 and net in single input 256, 56, 56
            tk::dnn::Layer    *route_s3_t1_root_layers[2] = { last2, last1 };
            tk::dnn::Route    route_s3_t1_root(&net, route_s3_t1_root_layers, 2);
            tk::dnn::Conv2d s3_t1_root_conv1(&net, 256, 1, 1, 1, 1, 0, 0, s3_t1_root_conv1_bin, true);
            tk::dnn::Activation   s3_t1_root_relu(&net, CUDNN_ACTIVATION_RELU);
    
    last1 = &s3_t1_root_relu;
    last3 = &s3_t1_root_relu;
    // tree 2
        // tree 1
            tk::dnn::Conv2d s3_t2_t1_conv1(&net, 256, 3, 3, 1, 1, 1, 1, s3_t2_t1_conv1_bin, true);
            tk::dnn::Activation s3_t2_t1_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d s3_t2_t1_conv2(&net, 256, 3, 3, 1, 1, 1, 1, s3_t2_t1_conv2_bin, true);
            tk::dnn::Shortcut s3_t2_t1_s1(&net, last1);
            tk::dnn::Activation   s3_t2_t1_relu(&net, CUDNN_ACTIVATION_RELU);

        last1 = &s3_t2_t1_relu;
        
        // tree 2
            tk::dnn::Conv2d s3_t2_t2_conv1(&net, 256, 3, 3, 1, 1, 1, 1, s3_t2_t2_conv1_bin, true);
            tk::dnn::Activation s3_t2_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
        
            tk::dnn::Conv2d s3_t2_t2_conv2(&net, 256, 3, 3, 1, 1, 1, 1, s3_t2_t2_conv2_bin, true);
            
            tk::dnn::Shortcut s3_t2_t2_s1(&net, last1);
            tk::dnn::Activation   s3_t2_t2_relu(&net, CUDNN_ACTIVATION_RELU);
            last2 = &s3_t2_t2_relu;

        // root
            // join last1 and net in single input 256, 56, 56
            tk::dnn::Layer    *route_s3_t2_root_layers[4] = { last2, last1, last4, last3};
            tk::dnn::Route    route_s3_t2_root(&net, route_s3_t2_root_layers, 4);
            tk::dnn::Conv2d s3_t2_root_conv1(&net, 256, 1, 1, 1, 1, 0, 0, s3_t2_root_conv1_bin, true);
            tk::dnn::Activation   s3_t2_root_relu(&net, CUDNN_ACTIVATION_RELU);
    
    last1 = &s3_t2_root_relu;
    // level 4
    // tree 1
        tk::dnn::Conv2d s4_t1_conv1(&net, 512, 3, 3, 2, 2, 1, 1, s4_t1_conv1_bin, true);
        tk::dnn::Activation s4_t1_relu1(&net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d s4_t1_conv2(&net, 512, 3, 3, 1, 1, 1, 1, s4_t1_conv2_bin, true);
        last2 = &s4_t1_conv2;

        // get the basicblock input and apply maxpool conv2d and relu
        tk::dnn::Layer    *route_s4_t1_layers[1] = { last1 };
        tk::dnn::Route    route_s4_t1(&net, route_s4_t1_layers, 1);
        // downsample
        tk::dnn::Pooling s4_t1_maxpool1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
        last4 = &s4_t1_maxpool1;
        // project
        tk::dnn::Conv2d s4_t1_residual1_conv1(&net, 512, 1, 1, 1, 1, 0, 0, s4_t1_project, true);
        
        tk::dnn::Shortcut s4_t1_s1(&net, last2);
        tk::dnn::Activation   s4_t1_relu(&net, CUDNN_ACTIVATION_RELU);

    last1 = &s4_t1_relu;
    
    // tree 2
        tk::dnn::Conv2d s4_t2_conv1(&net, 512, 3, 3, 1, 1, 1, 1, s4_t2_conv1_bin, true);
        tk::dnn::Activation s4_t2_relu1(&net, CUDNN_ACTIVATION_RELU);       
       
        tk::dnn::Conv2d s4_t2_conv2(&net, 512, 3, 3, 1, 1, 1, 1, s4_t2_conv2_bin, true);
        
        tk::dnn::Shortcut s4_t2_s1(&net, last1);
        tk::dnn::Activation   s4_t2_relu(&net, CUDNN_ACTIVATION_RELU);
        last2 = &s4_t2_relu;

    // root
        // join last1 and net in single input 128, 56, 56
        tk::dnn::Layer    *route_s4_root_layers[3] = { last2, last1, last4 };
        tk::dnn::Route    route_s4_root(&net, route_s4_root_layers, 3);
        tk::dnn::Conv2d s4_root_conv1(&net, 512, 1, 1, 1, 1, 0, 0, s4_root_conv1_bin, true);
        tk::dnn::Activation   s4_root_relu(&net, CUDNN_ACTIVATION_RELU);

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
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("dla34"));

    
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
