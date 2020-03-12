#ifndef YOLODETECTION_H
#define YOLODETECTION_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"

namespace tk { namespace dnn {

/**
 * 
 * @author Francesco Gatti
 */
class Yolo3Detection {

    private:
        tk::dnn::NetworkRT *netRT = nullptr;
        tk::dnn::Yolo* yolo[3];
        dnnType *input, *input_d;

        int ndets = 0;
        tk::dnn::Yolo::detection *dets = nullptr;

        cv::Mat imageF;
        cv::Mat bgr[3]; 

    public:
        int classes = 0;
        int num = 0;
        int n_masks = 0;
        float thresh = 0.3;
        cv::Scalar colors[256];

        // this is filled with results
        std::vector<tk::dnn::box> detected;

        // keep track of inference times (ms)
        std::vector<double> stats;

        Yolo3Detection() {}

        virtual ~Yolo3Detection() {}

        /**
         * Method used for inizialize the class
         * 
         * @return Success of the initialization
         */
        bool init(std::string tensor_path);
        cv::Mat draw(cv::Mat &frame);
        void update(cv::Mat &frame);

        tk::dnn::Yolo* getYoloLayer(int n=0) {
            if(n<3)
                return yolo[n];
            else 
                return nullptr;
        }

};

}}

#endif /* YOLODETECTION_H*/
