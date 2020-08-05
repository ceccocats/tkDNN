#ifndef Yolo3Detection_H
#define Yolo3Detection_H
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"

#include "DetectionNN.h"

namespace tk { namespace dnn { 

class Yolo3Detection : public DetectionNN
{
private:
    int num = 0;
    int nMasks = 0;
    int nDets = 0;
    tk::dnn::Yolo::detection *dets = nullptr;
    tk::dnn::Yolo* yolo[3];

    tk::dnn::Yolo* getYoloLayer(int n=0);

    cv::Mat bgr_h;
    
public:
    Yolo3Detection() {};
    ~Yolo3Detection() {}; 

    bool init(const std::string& tensor_path, const int n_classes=80, const int n_batches=1, const float conf_thresh=0.3);
    void preprocess(cv::Mat &frame, const int bi=0);
    void postprocess(const int bi=0,const bool mAP=false);
};


} // namespace dnn
} // namespace tk

#endif /* Yolo3Detection_H*/
