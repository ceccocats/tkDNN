#ifndef MOBILENETDETECTION_H
#define MOBILENETDETECTION_H

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"

#define N_COORDS 4


struct SSDSpec
{
    int feature_size = 0;
    int shrinkage = 0;
    int box_width = 0;
    int box_height = 0;
    int ratio1 = 0;
    int ratio2 = 0;

    SSDSpec() {}

    SSDSpec(int feature_size, int shrinkage, int box_width, int box_height, int ratio1, int ratio2) : feature_size(feature_size), shrinkage(shrinkage), box_width(box_width), box_height(box_height),
                                                                                                      ratio1(ratio1), ratio2(ratio2) {}

    void setAll(int feature_size, int shrinkage, int box_width, int box_height, int ratio1, int ratio2)
    {
        this->feature_size = feature_size;
        this->shrinkage = shrinkage;
        this->box_width = box_width;
        this->box_height = box_height;
        this->ratio1 = ratio1;
        this->ratio2 = ratio2;
    }

    void print()
    {
        std::cout << "fsize: " << feature_size << "\tshrinkage: " << shrinkage << "\t box W:" << box_width << "\tbox H: " << box_height << "\t x ratio:" << ratio1 << "\t y ratio:" << ratio2 << std::endl;
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

    int classes = 21;
    float iou_threshold = 0.45;
    float center_variance = 0.1;
    float size_variance = 0.2;
    float conf_thresh = 0.4;
    int input_h = 300;
    int input_w = 300;
    int image_size = 300;

    float *priors = nullptr;
    int n_priors = 0;

    cv::Mat origImg;
    cv::Mat bgr[3];

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
    void convert_locatios_to_boxes_and_center(float *priors, const int n_priors, float *locations, const float center_variance, const float size_variance);
    float iou(const tk::dnn::box &a, const tk::dnn::box &b);
    std::vector<tk::dnn::box> postprocess(float *locations, float *confidences, const int n_values, const float threshold, const int n_classes, const float iou_thresh, const int width, const int height);
    float get_color2(int c, int x, int max);

    cv::Scalar colors[256];
    std::vector<std::string> voc_class_name;

public:
    // keep track of inference times (ms)
    std::vector<double> stats;
    std::vector<tk::dnn::box> detected;

    MobilenetDetection() {}
    ~MobilenetDetection() {}

    void init(std::string tensor_path);
    cv::Mat draw();
    void update(cv::Mat &img);
};

} // namespace dnn
} // namespace tk


#endif /*MOBILENETDETECTION_H*/