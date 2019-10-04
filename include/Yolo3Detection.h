#ifndef YOLO3DDETECTION_H
#define YOLO3DDETECTION_H

#include <iostream>
#include <signal.h>
#include <stdlib.h> /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"

namespace tk
{
namespace dnn
{

/**
 * 
 * @author Francesco Gatti
 */
class Yolo3Detection
{

private:
    tk::dnn::NetworkRT *netRT = nullptr;
    tk::dnn::Yolo *yolo[3];
    dnnType *input, *input_d;

    int ndets = 0;
    tk::dnn::Yolo::detection *dets = nullptr;

    cv::Mat imageF;
    cv::Mat bgr[3];

public:
    int classes = 0;
    int num = 0;
    float thresh = 0.3;
    cv::Scalar colors[256];

    // this is filled with results
    std::vector<tk::dnn::box> detected;

    Yolo3Detection() {}

    virtual ~Yolo3Detection() {}

    /**
         * Method used to inizialize the class
         * 
         * @return Success of the initialization
         */
    bool init(std::string tensor_path);
    void addBorders(cv::Mat &imageORIG, cv::Mat &imageWBorders, int &top, int &left);
    void update(cv::Mat &frame);
};

} // namespace dnn
} // namespace tk

#endif /*YOLO3DDETECTION_H*/