#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Yolo3Detection.h"

bool gRun;

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
        input = argv[1]; 

    tk::dnn::Yolo3Detection yolo;
    yolo.init(net);

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::Mat frame;
    cv::Mat dnn_input;
    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    
    while(gRun) {
        cap >> frame; 
        if(!frame.data) {
            continue;
        }  
 
        // this will be resized to the net format
        dnn_input = frame.clone();
        // TODO: async infer
        yolo.update(dnn_input);

        // draw dets
        for(int i=0; i<yolo.detected.size(); i++) {
            tk::dnn::box b = yolo.detected[i];
            int x0   = b.x;
            int x1   = b.x + b.w;
            int y0   = b.y;
            int y1   = b.y + b.h;
            int obj_class = b.cl;
            float prob = b.prob;

            std::cout<<obj_class<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
            cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), yolo.colors[obj_class], 2);                
        }
    
        cv::imshow("detection", frame);
        cv::waitKey(1);
    }

    std::cout<<"detection end\n";   
    return 0;
}

