#ifndef MOBILENETDETECTION_H
#define MOBILENETDETECTION_H

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"

#include "tkdnn.h"

#define N_COORDS 4


struct SSDSpec
{
    int featureSize = 0;
    int shrinkage = 0;
    int boxWidth = 0;
    int boxHeight = 0;
    int ratio1 = 0;
    int ratio2 = 0;

    SSDSpec() {}

    SSDSpec(int feature_size, int shrinkage, int box_width, int box_height, int ratio1, int ratio2) : featureSize(feature_size), shrinkage(shrinkage), boxWidth(box_width), boxHeight(box_height),
                                                                                                      ratio1(ratio1), ratio2(ratio2) {}

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
        std::cout << "fsize: " << featureSize << "\tshrinkage: " << shrinkage << "\t box W:" << boxWidth << "\tbox H: " << boxHeight << "\t x ratio:" << ratio1 << "\t y ratio:" << ratio2 << std::endl;
    }
};


namespace tk
{
namespace dnn
{
class MobilenetDetection
{

private:
    tk::dnn::NetworkRT *netRT = nullptr;

    int classes;
    float IoUThreshold = 0.45;
    float centerVariance = 0.1;
    float sizeVariance = 0.2;
    float confThreshold = 0.4;
    int imageSize;

    float *priors = nullptr;
    int nPriors = 0;

    cv::Mat origImg;

    float *input, *input_d;
    float *locations_h, *confidences_h;

    tk::dnn::dataDim_t dim;

    dnnType *conf;
    dnnType *loc;

    float __colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
    int baseline = 0;
    float fontScale = 0.5;
    int thickness = 2;

    void generate_ssd_priors(const SSDSpec *specs, const int n_specs, bool clamp = true);
    void convert_locatios_to_boxes_and_center();
    float iou(const tk::dnn::box &a, const tk::dnn::box &b);
    void preprocess();
    std::vector<tk::dnn::box> postprocess(const int width, const int height);
    float get_color2(int c, int x, int max);

    cv::Scalar colors[256];
    std::vector<std::string> classesNames;

public:
    // keep track of inference times (ms)
    std::vector<double> stats;
    std::vector<tk::dnn::box> detected;

    MobilenetDetection() {}
    ~MobilenetDetection() {}

    void init(std::string tensor_path, int input_size, int n_classes);
    cv::Mat draw();
    void update(cv::Mat &img);
};

} // namespace dnn
} // namespace tk


#endif /*MOBILENETDETECTION_H*/