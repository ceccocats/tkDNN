#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>

#include "SegmentationNN.h"

bool gRun;
bool SAVE_RESULT = true;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);


    std::string net = "shelfnet_fp32.rt";
    if(argc > 1)
        net = argv[1]; 
    std::string input = "../../ShelfNet/ShelfNet18_realtime/data/leftImg8bit/test/modena/000302.png";
    if(argc > 2)
        input = argv[2]; 
    int n_batch = 1;
    if(argc > 3)
        n_batch = atoi(argv[3]); 
    bool show = false;
    if(argc > 4)
        show = atoi(argv[4]); 

    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    if(!show)
        SAVE_RESULT = true;

    int n_classes = 19;

    tk::dnn::SegmentationNN segNN;
    segNN.init(net, n_classes, n_batch);

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
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(1024, 1024));
    }

    cv::Mat frame;
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
        segNN.update(batch_dnn_input, n_batch);
        frame = segNN.draw();

        if(n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"segmentation end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(segNN.stats.begin(), segNN.stats.end())/n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(segNN.stats.begin(), segNN.stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<segNN.stats.size(); i++) mean += segNN.stats[i]; mean /= segNN.stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;   
    

    return 0;
}

