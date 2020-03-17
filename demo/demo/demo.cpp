#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
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


    char *net = "yolo3_berkeley.rt";
    if(argc > 1)
        net = argv[1]; 
    char *input = "../demo/yolo_test.mp4";
    if(argc > 2)
        input = argv[2]; 
    char ntype = 'y';
    if(argc > 3)
        ntype = argv[3][0]; 

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;    
    switch(ntype)
    {
        case 'y':
            yolo.init(net);
            break;
        case 'c':
            cnet.init(net);
            break;
        case 'm':
            mbnet.init(net, 512, 81);
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

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
    cv::Mat dnn_input;
    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    
    std::vector<tk::dnn::box> detected_bbox;

    while(gRun) {
        cap >> frame; 
        if(!frame.data) {
            break;
        }  
 
        // this will be resized to the net format
        dnn_input = frame.clone();
        // TODO: async infer
        switch(ntype)
        {
            case 'y':
                yolo.update(dnn_input);
                frame = yolo.draw(frame);
                break;
            case 'c':
                cnet.update(dnn_input);
                frame = cnet.draw(dnn_input);
                break;
            case 'm':
                mbnet.update(dnn_input);
                frame = mbnet.draw();
                break;
            default:
                FatalError("Network type not allowed!\n");
        }
                
        cv::imshow("detection", frame);
        cv::waitKey(1);
        if(SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";   
    double mean = 0; 
    switch(ntype)
    {
        case 'y':
            std::cout<<COL_GREENB<<"\n\nTime stats:\n";
            std::cout<<"Min: "<<*std::min_element(yolo.stats.begin(), yolo.stats.end())<<" ms\n";    
            std::cout<<"Max: "<<*std::max_element(yolo.stats.begin(), yolo.stats.end())<<" ms\n";    
            for(int i=0; i<yolo.stats.size(); i++) mean += yolo.stats[i]; mean /= yolo.stats.size();
            std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;  
            break;
        case 'c':
            std::cout<<COL_GREENB<<"\n\nTime stats:\n";
            std::cout<<"Min: "<<*std::min_element(cnet.stats.begin(), cnet.stats.end())<<" ms\n";    
            std::cout<<"Max: "<<*std::max_element(cnet.stats.begin(), cnet.stats.end())<<" ms\n";    
            for(int i=0; i<cnet.stats.size(); i++) mean += cnet.stats[i]; mean /= cnet.stats.size();
            std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;
            break;
        case 'm':
            std::cout<<COL_GREENB<<"\n\nTime stats:\n";
            std::cout<<"Min: "<<*std::min_element(mbnet.stats.begin(), mbnet.stats.end())<<" ms\n";    
            std::cout<<"Max: "<<*std::max_element(mbnet.stats.begin(), mbnet.stats.end())<<" ms\n";    
            for(int i=0; i<mbnet.stats.size(); i++) mean += mbnet.stats[i]; mean /= mbnet.stats.size();
            std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;   
            break;
        default:
            FatalError("Network type not allowed!\n");
    }

    return 0;
}

