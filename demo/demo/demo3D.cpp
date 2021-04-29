#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
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
    #ifdef __linux__ 
        std::string input = "../demo/yolo_test.mp4";
    #elif _WIN32
        std::string input = "..\\..\\..\\demo\\yolo_test.mp4";
    #endif

    if(argc > 2)
        input = argv[2]; 
    char ntype = 'c';
    if(argc > 3)
        ntype = argv[3][0]; 
    int n_classes = 3;
    if(argc > 4)
        n_classes = atoi(argv[4]); 
    int n_batch = 1;
    if(argc > 5)
        n_batch = atoi(argv[5]); 
    bool show = true;
    if(argc > 6)
        show = atoi(argv[6]); 
    float conf_thresh=0.3;
    if(argc > 7)
        conf_thresh = atof(argv[7]);     

    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

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

    detNN->init(net, n_classes, n_batch, conf_thresh);

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
    
    std::cout<<COL_GREENB<<"\n\nTime preprocessing stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->pre_stats.begin(), detNN->pre_stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->pre_stats.begin(), detNN->pre_stats.end())<<" ms\n";    
    for(int i=0; i<detNN->pre_stats.size(); i++) mean += detNN->pre_stats[i]; mean /= detNN->pre_stats.size();
    std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;   
    mean=0;
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;   
    mean=0;
    std::cout<<COL_GREENB<<"\n\nTime postprocessing stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->post_stats.begin(), detNN->post_stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(detNN->post_stats.begin(), detNN->post_stats.end())<<" ms\n";    
    for(int i=0; i<detNN->post_stats.size(); i++) mean += detNN->post_stats[i]; mean /= detNN->post_stats.size();
    std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;   
    
    
    return 0;
}

