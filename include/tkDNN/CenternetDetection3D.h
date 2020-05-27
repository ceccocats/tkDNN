#ifndef CENTERNETDETECTION3D_H
#define CENTERNETDETECTION3D_H

#include "kernels.h"
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort

#include "DetectionNN3D.h"

#include "kernelsThrust.h"


namespace tk { namespace dnn { 

class CenternetDetection3D : public DetectionNN3D
{
private:
    tk::dnn::dataDim_t dim;
    tk::dnn::dataDim_t dim2;
    tk::dnn::dataDim_t dim_hm; 
    tk::dnn::dataDim_t dim_wh; 
    tk::dnn::dataDim_t dim_reg;
    tk::dnn::dataDim_t dim_dep;
    tk::dnn::dataDim_t dim_rot;
    tk::dnn::dataDim_t dim_dim;
    float *topk_scores;
    int *topk_inds_;
    float *topk_ys_;
    float *topk_xs_;
    int *ids_d, *ids_;

    float *ones;

    float *scores, *scores_d;
    int *clses, *clses_d; 
    int *topk_inds_d;
    float *topk_ys_d; 
    float *topk_xs_d;
    int *inttopk_xs_d, *inttopk_ys_d;

    float *xs, *ys;

    float *dep, *rot, *dim_, *wh;
    float *dep_d, *rot_d, *dim_d, *wh_d;
    
    float *target_coords;

    #ifdef OPENCV_CUDACONTRIB
        float *mean_d;
        float *stddev_d;
    #else
        cv::Vec<float, 3> mean;
        cv::Vec<float, 3> stddev;
        dnnType *input;
    #endif
    cv::Mat r;
    cv::Mat calibs;
    float *d_ptrs;

    cv::Mat src;
    cv::Mat dst;
    cv::Mat dst2;  
    cv::Mat trans, trans2;
    //processing
    int K = 100;
    int width = 128;//56;        // TODO

    // pointer used in the kernels
    float *src_out;
    int *ids_out;

    struct threshold op;
    float peakThreshold = 0.2;
    float centerThreshold = 0.3; //default 0.5
    cv::Mat corners, pts3DHomo;  

    std::vector<box3D> detected3D;
    std::vector<int>cls3D;
    std::vector<std::vector<int>> face_id;

public:
    CenternetDetection3D() {};
    ~CenternetDetection3D() {}; 

    bool init(const std::string& tensor_path, const int n_classes=3);
    void preprocess(cv::Mat &frame);
    void postprocess();
    cv::Mat draw(cv::Mat &frame);
};


} // namespace dnn
} // namespace tk


#endif /*CENTERNETDETECTION_H*/