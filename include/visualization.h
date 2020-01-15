#ifndef VIZUALIZATION_H
#define VIZUALIZATION_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//saliency
#include <opencv2/core/utility.hpp>
//#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>
#include <cstring>

#include "tracker.h"
#include "cameraUtils.h"
#include "calibration.h"
#include "boxDetection.h"

struct Show_t
{
    cv::Mat original, detection, topview, disparity;
    bool update_o, update_de, update_t, update_di;
    // a single mutex for each operation - the show_updates function must get all mutex
    std::mutex mutex_o, mutex_de, mutex_t, mutex_di;
};

extern Show_t updates;
extern bool gRun;
extern std::string obj_class[10];

/* Thread function to show the updated images 
**/
void *show_updates(void *x_void_ptr);
void *originalFrame(void *x_void_ptr);
void *detectionFrame(void *x_void_ptr);
void *topviewFrame(void *x_void_ptr);
void *disparityFrame(void *x_void_ptr);

#endif /*VIZUALIZATION_H*/