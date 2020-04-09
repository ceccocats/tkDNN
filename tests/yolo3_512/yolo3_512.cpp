#include<iostream>
#include<vector>
#include "tkdnn.h"

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 512, 512, 1);
    tk::dnn::Network net(dim);

    // create yolo3 model
    std::string bin_path  = "../tests/yolo3_512";
    downloadWeightsifDoNotExist("../tests/yolo3_512/layers/input.bin", bin_path, "https://cloud.hipert.unimore.it/s/e7HfScx77JEHeYb/download");
    int classes = 80;
    tk::dnn::Yolo *yolo [3];
    #include "models/Yolo3.h"

    

    // fill classes names
    for(int i=0; i<3; i++) {
        yolo[i]->classesNames = {"person" , "bicycle" , "car" , "motorbike" , "aeroplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "backpack" , "umbrella" , "handbag" , "tie" , "suitcase" , "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , "banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" , "pizza" , "donut" , "cake" , "chair" , "sofa" , "pottedplant" , "bed" , "diningtable" , "toilet" , "tvmonitor" , "laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" , "oven" , "toaster" , "sink" , "refrigerator" , "book" , "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush"};
    }

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    
    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("yolo3_512"));

    // the network have 3 outputs
    tk::dnn::dataDim_t out_dim[3];
    for(int i=0; i<3; i++) out_dim[i] = yolo[i]->output_dim; 
    dnnType *cudnn_out[3], *rt_out[3]; 

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TIMER_START
        net.infer(dim1, data);    
        TIMER_STOP
        dim1.print();   
    }
    for(int i=0; i<3; i++) cudnn_out[i] = yolo[i]->dstData;
    
    printCenteredTitle(" compute detections ", '=', 30);
    TIMER_START
    int ndets = 0;
    tk::dnn::Yolo::detection *dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);
    for(int i=0; i<3; i++) yolo[i]->computeDetections(dets, ndets, net.input_dim.w, net.input_dim.h, 0.5);
    tk::dnn::Yolo::mergeDetections(dets, ndets, classes);

    for(int j=0; j<ndets; j++) {
        tk::dnn::Yolo::box b = dets[j].bbox;
        int x0   = (b.x-b.w/2.);
        int x1   = (b.x+b.w/2.);
        int y0   = (b.y-b.h/2.);
        int y1   = (b.y+b.h/2.);

        int cl = 0;
        for(int c = 0; c < classes; ++c){
            float prob = dets[j].prob[c];
            if(prob > 0)
                cl = c;
        }
        std::cout<<cl<<": "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
    }
    TIMER_STOP
    
    tk::dnn::dataDim_t dim2 = dim;
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
