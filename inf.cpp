
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#ifdef __linux__
#include <unistd.h>
#endif

#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Yolo3Detection.h"
//#include "CenternetDetection.h"
//#include "MobilenetDetection.h"
#include "evaluation.h"
#include <chrono>
#include <cstdint>
#include <iostream>

uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
int baggage() {
    std::cout << timeSinceEpochMillisec() << std::endl;
    char ntype = 'y';
    const char *config_filename = "../demo/config.yaml";
    const char * net = "../demo/yolo4_fp32.rt";
    const char * img_path = "../demo/demo.jpg";
    bool show = false;
    bool verbose;
    int classes, map_points, map_levels;
    float map_step, IoU_thresh, conf_thresh;

    //read parameters
    tk::dnn::readmAPParams(config_filename, classes,  map_points, map_levels, map_step,
                IoU_thresh, conf_thresh, verbose);


    //extract network name from rt path
    std::string net_name;
    removePathAndExtension(net, net_name);
    std::cout<<"Network: "<<net_name<<std::endl;

    //open files (if needed)
    std::ofstream times, memory, coco_json;

    // instantiate detector
    tk::dnn::Yolo3Detection yolo;
 //   tk::dnn::CenternetDetection cnet;
//    tk::dnn::MobilenetDetection mbnet;
    tk::dnn::DetectionNN *detNN;  
    int n_classes = classes;
//    float conf_threshold=0.001;
    detNN = &yolo;
    detNN->init(net, n_classes, 1, conf_thresh);

    //read images 
   // std::ifstream all_labels(labels_path);
    std::cout << timeSinceEpochMillisec() << std::endl;
    std::string l_filename;
    std::vector<tk::dnn::Frame> images;
    std::vector<tk::dnn::box> detected_bbox;

    std::cout<<"Reading groundtruth and generating detections"<<std::endl;

    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

//    int images_done;
 //   for (images_done=0 ; std::getline(all_labels, l_filename) && images_done < n_images ; ++images_done) {
   //     std::cout <<COL_ORANGEB<< "Images done:\t" << images_done<< "\n"<<COL_END;

    tk::dnn::Frame f;
   // f.lFilename = l_filename;
   // f.iFilename = l_filename;
    //convertFilename(f.iFilename, "labels", "images", ".txt", ".jpg");

    // read frame
    //if(!fileExist(f.iFilename.c_str()))
      //  FatalError("Wrong image file path.");
    cv::Mat frame = cv::imread(img_path, cv::IMREAD_COLOR);
    std::vector<cv::Mat> batch_frames;
    batch_frames.push_back(frame);
    int height = frame.rows;
    int width = frame.cols;

//    if(!frame.data) 
 //       break;
    std::vector<cv::Mat> batch_dnn_input;
    batch_dnn_input.push_back(frame.clone());
    std::cout<<"test1"<<"\n";
    //inference 
    detected_bbox.clear();
    detNN->update(batch_dnn_input,1);
    detNN->draw(batch_frames);
    detected_bbox = detNN->detected;
    std::cout<<"test2"<<"\n";
    // save detections labels
    for(auto d:detected_bbox){
        //convert detected bb in the same format as label
        //<x_center>/<image_width> <y_center>/<image_width> <width>/<image_width> <height>/<image_width>
        tk::dnn::BoundingBox b;
        b.x = (d.x + d.w/2) / width;
        b.y = (d.y + d.h/2) / height;
        b.w = d.w / width;
        b.h = d.h / height;
        b.prob = d.prob;
        b.cl = d.cl;
        f.det.push_back(b);

        std::cout<< d.cl << " "<< d.prob << " "<< b.x << " "<< b.y << " "<< b.w << " "<< b.h <<"\n";

        if(show)// draw rectangle for detection
            cv::rectangle(batch_frames[0], cv::Point(d.x, d.y), cv::Point(d.x + d.w, d.y + d.h), cv::Scalar(0, 0, 255), 2);             
    }
    //images.push_back(f);
    
    if(show){
        cv::imshow("detection", batch_frames[0]);
        cv::waitKey(0);
    }
    std::cout << timeSinceEpochMillisec() << std::endl;
    return 0;
    }

