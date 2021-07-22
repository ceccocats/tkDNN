#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "demo_utils.h"
#include "CenterTrack.h"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);


    std::string net = "dla34_cnet3d_track_fp32.rt";
    if(argc > 1)
        net = argv[1]; 
    #ifdef __linux__ 
        std::string input = "../demo/yolo_test.mp4";
    #elif _WIN32
        std::string input = "..\\..\\..\\demo\\yolo_test.mp4";
    #endif

    if(argc > 2)
        input = argv[2]; 
    std::string calib_params = "";
    if(argc > 3)
        calib_params = argv[3];
    char ntype = 'c';
    if(argc > 4)
        ntype = argv[4][0]; 
    int n_classes = 3;
    if(argc > 5)
        n_classes = atoi(argv[5]); 
    int n_batch = 1;
    if(argc > 6)
        n_batch = atoi(argv[6]); 
    bool show = true;
    if(argc > 7)
        show = atoi(argv[7]); 
    float conf_thresh=0.3;
    if(argc > 8)
        conf_thresh = atof(argv[8]);   
    bool t3d = true;
    if(argc > 9)
        t3d = atoi(argv[9]);
    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    if(!show)
	    SAVE_RESULT = true;

    tk::dnn::CenterTrack ctrack;

    tk::dnn::TrackingNN *trackNN;  

    switch(ntype)
    {
        case 'c':
            trackNN = &ctrack;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }
    std::vector<cv::Mat> calibs;
    if(!calib_params.empty() && calib_params!="NULL") {
        std::cout<<"calib_params: "<<calib_params<<std::endl;
        cv::Mat calib;
        // the calibration matrix must be a 3x3 matrix
        readCalibrationMatrix(calib_params, calib);
        for(int bi=0; bi< n_batch; ++bi)
            calibs.push_back(calib);
    }
    trackNN->init(net, n_classes, n_batch, conf_thresh, t3d, calibs);

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
        trackNN->update(batch_dnn_input, n_batch, false, nullptr, false);
        trackNN->draw(batch_frame);

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
    std::cout<<"Min: "<<*std::min_element(trackNN->pre_stats.begin(), trackNN->pre_stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(trackNN->pre_stats.begin(), trackNN->pre_stats.end())<<" ms\n";    
    for(int i=0; i<trackNN->pre_stats.size(); i++) mean += trackNN->pre_stats[i]; mean /= trackNN->pre_stats.size();
    std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;   
    mean=0;
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(trackNN->stats.begin(), trackNN->stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(trackNN->stats.begin(), trackNN->stats.end())<<" ms\n";    
    for(int i=0; i<trackNN->stats.size(); i++) mean += trackNN->stats[i]; mean /= trackNN->stats.size();
    std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;   
    mean=0;
    std::cout<<COL_GREENB<<"\n\nTime postprocessing stats:\n";
    std::cout<<"Min: "<<*std::min_element(trackNN->post_stats.begin(), trackNN->post_stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(trackNN->post_stats.begin(), trackNN->post_stats.end())<<" ms\n";    
    for(int i=0; i<trackNN->post_stats.size(); i++) mean += trackNN->post_stats[i]; mean /= trackNN->post_stats.size();
    std::cout<<"Avg: "<<mean<<" ms\n"<<COL_END;   
    
    
    return 0;
}

