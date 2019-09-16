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

bool 			gRun;
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

    tk::dnn::Yolo3Detection yolo;
    yolo.init(net);

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";


    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        int h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", CV_FOURCC('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    cv::Mat dnn_input;
    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    
    while(gRun) {
        cap >> frame; 
        if(!frame.data) {
            break;
        }  
 
        // this will be resized to the net format
        dnn_input = frame.clone();
        // TODO: async infer
        yolo.update(dnn_input);

        // draw dets
        for(int i=0; i<yolo.detected.size(); i++) {
            tk::dnn::box b = yolo.detected[i];
            int x0   				= b.x;
            int x1   				= b.x + b.w;
            int y0   				= b.y;
            int y1   				= b.y + b.h;
            std::string det_class 	= yolo.getYoloLayer()->classesNames[b.cl];
            float prob 				= b.prob;

            std::cout<<det_class<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
			// draw rectangle
            cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), yolo.colors[b.cl], 2); 

	        // draw label
            int baseline = 0;
            float fontScale = 0.5;
            int thickness = 2;
            cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
            cv::rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + textSize.width - 2), (y0 - textSize.height - 2)), yolo.colors[b.cl], -1);                      
            cv::putText(frame, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
        }
    
        cv::imshow("detection", frame);
        cv::waitKey(1);
        if(SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";   
    return 0;
}

