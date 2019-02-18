#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tkDNN/tkdnn.h>

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
        static const int classes = 10;
        static const int num = 3;
        float thresh = 0.3;
        cv::Scalar colors[classes];

        // this is filled with results
        std::vector<tk::dnn::box> detected;

        Yolo3Detection() {}

        virtual ~Yolo3Detection() {}

        /**
         * Method used for inizialize the class
         * 
         * @return Success of the initialization
         */
        bool init(std::string tensor_path);

        void update(cv::Mat &frame);

};
