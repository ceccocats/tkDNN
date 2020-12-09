#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>

#include "CenternetDetection3D.h"
#include "CenternetDetection3DTrack.h"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);


    std::string net = "dla34_cnet3d_fp32.rt";
    if(argc > 1)
        net = argv[1]; 
    std::string input = "../demo/yolo_test.mp4";
    if(argc > 2)
        input = argv[2]; 
    char ntype = 'c';
    if(argc > 3)
        ntype = argv[3][0]; 
    int n_classes = 3;
    if(argc > 4)
        n_classes = atoi(argv[4]); 
    bool show = false;
    if(argc > 5)
        show = atoi(argv[5]);

    if(!show)
	SAVE_RESULT = true;

    tk::dnn::CenternetDetection3D cnet;
    tk::dnn::CenternetDetection3DTrack ctrack;

    tk::dnn::DetectionNN3D *detNN;  

    switch(ntype)
    {
        case 'c':
            detNN = &cnet;
            break;
        case 't':
            detNN = &ctrack;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

    detNN->init(net, n_classes);

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
    if(show)
	cv::namedWindow("detection", cv::WINDOW_NORMAL);
    
    std::vector<tk::dnn::box> detected_bbox;

    while(gRun) {
        cap >> frame; 
        if(!frame.data) {
            break;
        }  
 
        // this will be resized to the net format
        dnn_input = frame.clone();
        
        //inference
        detNN->update(dnn_input);
        frame = detNN->draw(frame);
	
	if(show) {
	    cv::imshow("detection", frame);
            cv::waitKey(1);
	}
        if(SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;   
    

    return 0;
}

