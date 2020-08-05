#ifndef MOBILENETDETECTION_H
#define MOBILENETDETECTION_H

#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

#include "DetectionNN.h"

#define N_COORDS 4
#define N_SSDSPEC 6

namespace tk { namespace dnn { 

struct SSDSpec
{
    int featureSize = 0;
    int shrinkage = 0;
    int boxWidth = 0;
    int boxHeight = 0;
    int ratio1 = 0;
    int ratio2 = 0;

    SSDSpec() {}
    SSDSpec(int feature_size, int shrinkage, int box_width, int box_height, int ratio1, int ratio2) : 
                    featureSize(feature_size), shrinkage(shrinkage), boxWidth(box_width), 
                    boxHeight(box_height), ratio1(ratio1), ratio2(ratio2) {}
    void setAll(int feature_size, int shrinkage, int box_width, int box_height, int ratio1, int ratio2)
    {
        this->featureSize = feature_size;
        this->shrinkage = shrinkage;
        this->boxWidth = box_width;
        this->boxHeight = box_height;
        this->ratio1 = ratio1;
        this->ratio2 = ratio2;
    }
    void print()
    {
        std::cout << "fsize: " << featureSize << "\tshrinkage: " << shrinkage << 
                    "\t box W:" << boxWidth << "\tbox H: " << boxHeight << 
                    "\t x ratio:" << ratio1 << "\t y ratio:" << ratio2 << std::endl;
    }
};

class MobilenetDetection : public DetectionNN
{
private:
    float IoUThreshold = 0.45;
    float centerVariance = 0.1;
    float sizeVariance = 0.2;
    int imageSize;

    float *priors = nullptr;
    int nPriors = 0;
    float *locations_h, *confidences_h;

    

    void generate_ssd_priors(const SSDSpec *specs, const int n_specs, bool clamp = true);
    void convert_locatios_to_boxes_and_center();
    float iou(const tk::dnn::box &a, const tk::dnn::box &b);

    

public:
    MobilenetDetection() {};
    ~MobilenetDetection() {}; 

    bool init(const std::string& tensor_path, const int n_classes, const int n_batches=1, const float conf_thresh=0.3);
    void preprocess(cv::Mat &frame, const int bi=0);
    void postprocess(const int bi=0,const bool mAP=false);
};


} // namespace dnn
} // namespace tk

#endif /*MOBILENETDETECTION_H*/