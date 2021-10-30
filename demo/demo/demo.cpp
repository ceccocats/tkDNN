#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);


    std::string net = "yolo4tiny_fp32.rt";
    #ifdef __linux__
        std::string cfgPath = "../tests/darknet/cfg/yolo4tiny.cfg";
    #elif _WIN32
         std::string cfgPath = "..\\tests\\darknet\\cfg\\yolo4tiny.cfg";
    #endif

    #ifdef __linux__
             std::string namePath = "../tests/darknet/names/coco.names";
    #elif _WIN32
            std::string namePath = "..\\tests\\darknet\\names\\coco.names";
    #endif

    if(argc > 1)
        net = argv[1]; 
    #ifdef __linux__ 
        std::string input = "../demo/yolo_test.mp4";
    #elif _WIN32
        std::string input = "..\\demo\\yolo_test.mp4";
    #endif

    char ntype = 'y';
    if(argc > 2)
        ntype = argv[2][0];
    int n_classes = 80;
    if(argc > 3)
        n_classes = atoi(argv[3]);
    if(argc > 4)
        cfgPath = argv[4];
    if(argc > 5)
        namePath = argv[5];
    if(argc > 6)
        input = argv[6];
    int n_batch = 1;
    if(argc > 7)
        n_batch = atoi(argv[7]);
    bool show = true;
    if(argc > 8)
        show = atoi(argv[8]);
    float conf_thresh=0.3;
    if(argc >= 9)
        conf_thresh = atof(argv[9]);

    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    if(!show)
        SAVE_RESULT = true;

    if(ntype == 'c' || ntype == 'm'){
        cfgPath = "";
        namePath = "";

    }
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;

    tk::dnn::DetectionNN *detNN;  

    switch(ntype)
    {
        case 'y':
            detNN = &yolo;
            break;
        case 'c':
            detNN = &cnet;
            break;
        case 'm':
            detNN = &mbnet;
            n_classes++;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

    detNN->init(net,cfgPath,namePath,n_classes,n_batch,conf_thresh);

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    while(gRun) {
        batch_dnn_input.clear();
        batch_frame.clear();
        
        for(int bi=0; bi< n_batch; ++bi){
            cap >> frame; 
            if(!frame.data) 
                break;
            
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        } 
        if(!frame.data) 
            break;
    
        //inference
        detNN->update(batch_dnn_input, n_batch);
        detNN->draw(batch_frame);

        if(show){
            for(int bi=0; bi< n_batch; ++bi){
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
            }
        }
        if(n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
   std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;   
    

    return 0;
}

