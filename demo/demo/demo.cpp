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

    tk::dnn::Yolo3Detection yolo;
    yolo.init("yolo3_berkeley");

    gRun = true;

    char *input = "../demo/yolo_test.mp4";
    if(argc > 1)
        input = argv[1]; 

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::Mat frame;
    cv::Mat dnn_input;
    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("detection", 544*1.2, 320*1.2);
    
    while(gRun) {
        cap >> frame; 
        if(!frame.data) {
            continue;
        }  
 
        dnn_input = frame.clone();
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

