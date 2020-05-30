#include<iostream>
#include<vector>
#include "tkdnn.h"
#include "DarknetParser.h"

int main() {

    // create yolo3 model
    std::string bin_path  = "yolo3";
    downloadWeightsifDoNotExist("yolo3/layers/input.bin", bin_path, "https://cloud.hipert.unimore.it/s/jPXmHyptpLoNdNR/download");
    
    tk::dnn::Network *net = tk::dnn::darknetParser("../tests/yolo3/yolov3.cfg", "yolo3/layers");
    net->print();

    std::vector<tk::dnn::Yolo*> yolo;
    for(int i=0; i<net->num_layers; i++) {
        if(net->layers[i]->getLayerType() == tk::dnn::layerType_t::LAYER_YOLO)
            yolo.push_back((tk::dnn::Yolo*)net->layers[i]);
    }

    // fill classes names
    for(int i=0; i<3; i++) {
        yolo[i]->classesNames = {"person" , "bicycle" , "car" , "motorbike" , "aeroplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "backpack" , "umbrella" , "handbag" , "tie" , "suitcase" , "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , "banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" , "pizza" , "donut" , "cake" , "chair" , "sofa" , "pottedplant" , "bed" , "diningtable" , "toilet" , "tvmonitor" , "laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" , "oven" , "toaster" , "sink" , "refrigerator" , "book" , "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush"};
    }

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(net, net->getNetworkRTName("yolo3"));
    

    std::string input_bin = bin_path + "/layers/input.bin";
    std::vector<std::string> output_bins = {
        bin_path + "/debug/layer82_out.bin",
        bin_path + "/debug/layer94_out.bin",
        bin_path + "/debug/layer106_out.bin"
    };

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, net->input_dim.tot(), &input_h, &data);

    // the network have 3 outputs
    tk::dnn::dataDim_t out_dim[3];
    for(int i=0; i<3; i++) out_dim[i] = yolo[i]->output_dim; 
    dnnType *cudnn_out[3], *rt_out[3]; 

    tk::dnn::dataDim_t dim1 =  net->input_dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TIMER_START
        net->infer(dim1, data);    
        TIMER_STOP
        dim1.print();   
    }
    for(int i=0; i<3; i++) cudnn_out[i] = yolo[i]->dstData;

    
    tk::dnn::dataDim_t dim2 = net->input_dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }
    for(int i=0; i<3; i++) rt_out[i] = (dnnType*)netRT.buffersRT[i+1];

    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0; 
    for(int i=0; i<3; i++) {
        printCenteredTitle((std::string(" YOLO ") + std::to_string(i) + " CHECK RESULTS ").c_str(), '=', 30);
        dnnType *out, *out_h;
        int odim = out_dim[i].tot();
        readBinaryFile(output_bins[i], odim, &out_h, &out);
        std::cout<<"CUDNN vs correct"; 
        ret_cudnn |= checkResult(odim, cudnn_out[i], out) == 0 ? 0: ERROR_CUDNN;
        std::cout<<"TRT   vs correct"; 
        ret_tensorrt |= checkResult(odim, rt_out[i], out) == 0 ? 0 : ERROR_TENSORRT;
        std::cout<<"CUDNN vs TRT    "; 
        ret_cudnn_tensorrt |= checkResult(odim, cudnn_out[i], rt_out[i]) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    }
    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
