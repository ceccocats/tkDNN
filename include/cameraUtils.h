#ifndef CAMERAUTILS_H
#define CAMERAUTILS_H

#include <vector>
#include <mutex>
#include <opencv2/core/core.hpp>
#include "tracker.h"
#include "Yolo3Detection.h"

struct Camera_t
{
    int CAM_IDX;
    char *input;
    char *pmatrix;
    char *maskfile;
    char *cameraCalib;
    char *maskFileOrient;
    bool to_show;
    tk::dnn::Yolo3Detection *yolo;
    double adfGeoTransform[6];
};

struct Frame_t
{
    char *input;
    cv::Mat frame;
    int frame_nbr;
    // sem_vc for mainthread, videocapturethread, originalthread and disparitythread
    std::mutex sem_vc;
};

struct ModFrame_t
{
    std::vector<Tracker> trackers;
    geodetic_converter::GeodeticConverter gc;
    double adfGeoTransform[6];
    cv::Mat H;
    cv::Mat original_frame;
    tk::dnn::Yolo3Detection yolo;
    cv::Mat mask;
    // sem for mainthread, detectionthread and topviewthread
    std::mutex sem;
};

#endif /*CAMERAUTILS_H*/