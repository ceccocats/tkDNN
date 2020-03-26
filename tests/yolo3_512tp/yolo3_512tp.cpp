#include<iostream>
#include<vector>
#include "tkdnn.h"

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 512, 512, 1);
    tk::dnn::Network net(dim);

    // create yolo3 model
    std::string bin_path  = "../tests/yolo3_512tp";
    // downloadWeightsifDoNotExist("../tests/yolo3_512tp/layers/input.bin", bin_path, );
    int classes = 3;
    tk::dnn::Yolo *yolo [3];
    #include "models/Yolo3.h"

    // fill classes names
    for(int i=0; i<3; i++) {
        yolo[i]->classesNames = {"Dent", "Wrinkle", "UnsealedFlaps"};
    }

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    
    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, "yolo3_512tp.rt");

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

    for(int i=0; i<3; i++) {
        printCenteredTitle((std::string(" YOLO ") + std::to_string(i) + " CHECK RESULTS ").c_str(), '=', 30);
        dnnType *out, *out_h;
        int odim = out_dim[i].tot();
        readBinaryFile(output_bins[i], odim, &out_h, &out);
        std::cout<<"CUDNN vs correct"; checkResult(odim, cudnn_out[i], out);
        std::cout<<"TRT   vs correct"; checkResult(odim, rt_out[i], out);
        std::cout<<"CUDNN vs TRT    "; checkResult(odim, cudnn_out[i], rt_out[i]);
    }
    return 0;
}
