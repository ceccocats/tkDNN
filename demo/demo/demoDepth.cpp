#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "tkDNN/DepthNN.h"

bool gRun;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, sig_handler);

    std::string net = "monodepth2_fp32.rt";
    if(argc > 1)
        net = argv[1]; 
    #ifdef __linux__ 
        std::string input = "../demo/yolo_test.mp4";
    #elif _WIN32
        std::string input = "..\\..\\..\\demo\\yolo_test.mp4";
    #endif
    if(argc > 2)
        input = argv[2]; 
    bool show = true;
    if(argc > 3)
        show = atoi(argv[3]); 
    bool save = true;
    if(argc > 4)
        save = atoi(argv[4]); 

    std::cout   <<"Net settings - net: "<< net
                <<"\n"; 
    std::cout   <<"Demo settings - input: "<< input
                <<", show: "<< show
                <<", save: "<< save<<"\n\n"; 
    
    tk::dnn::DepthNN depthNN;  

    // create depth network 
    int n_batch = 1;
    depthNN.init(net, n_batch);

    // open video stream
    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::VideoWriter resultVideo;
    if(save) {
        int w = depthNN.output_w;
        int h = depthNN.output_h;
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    if(show)
        cv::namedWindow("depth", cv::WINDOW_NORMAL);

    cv::Mat frame;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    // start detection loop
    gRun = true;
    while(gRun) {
        batch_dnn_input.clear();
        batch_frame.clear();

        //read frame
        cap >> frame; 
        if(!frame.data) 
            break;
        batch_frame.push_back(frame);
        batch_dnn_input.push_back(frame.clone());
    
        //inference
        depthNN.update(batch_dnn_input, 1);
        if(show){
            cv::imshow("depth", depthNN.depthMats[0]);
            cv::waitKey(1);

        }

        if(save)
            resultVideo << depthNN.depthMats[0];
    }

    std::cout<<"detection end\n";   
    
    double mean = 0; 
    std::cout<<COL_GREENB<<"\n\nTime stats depth:\n";
    std::cout<<"Min: "<<*std::min_element(depthNN.stats.begin(), depthNN.stats.end())<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(depthNN.stats.begin(), depthNN.stats.end())<<" ms\n";    
    for(int i=0; i<depthNN.stats.size(); i++) mean += depthNN.stats[i]; mean /= depthNN.stats.size();
    std::cout<<"Avg: "<<mean<<" ms\t"<<1000/(mean)<<" FPS\n";   

    return 0;
}

