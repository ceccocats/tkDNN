#include <iostream>
#include <cstring>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include "utils.h"
#include <time.h>
#include "kernels.h"
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"
#include "sorting.h"

namespace tk { namespace dnn {

/**
 * 
 * @author Francesco Gatti
 */
class CenternetDetection {

    private:
        tk::dnn::NetworkRT *netRT = nullptr;
        dnnType *input_h, *input, *input_d;

        int ndets = 0;
        // tk::dnn::Yolo::detection *dets = nullptr;

        cv::Mat imageF;
        cv::Mat bgr[3]; 

        // variable to test cnet on dog pictures
        tk::dnn::dataDim_t dim;
        tk::dnn::dataDim_t dim2;
        cv::Size sz;
        const char *input_bin = "../tests/resnet101_cnet/debug/input.bin";

        // pre-process
        tk::dnn::dataDim_t dim_hm; 
        tk::dnn::dataDim_t dim_wh; 
        tk::dnn::dataDim_t dim_reg;
        float *topk_scores;
        int *topk_inds_;
        float *topk_ys_;
        float *topk_xs_;
        int *ids_d, *ids_, *ids_2, *ids_2d;

        float *scores, *scores_d;
        int *clses, *clses_d; 
        int *topk_inds_d;
        float *topk_ys_d; 
        float *topk_xs_d;
        int *inttopk_xs_d, *inttopk_ys_d;
    

        float *bbx0, *bby0, *bbx1, *bby1;
        float *bbx0_d, *bby0_d, *bbx1_d, *bby1_d;
        
        float *target_coords;
        
        cv::Vec<float, 3> mean;
        cv::Vec<float, 3> stddev;
        cv::Mat src;
        cv::Mat dst;  
        //processing
        float toll = 0.000001;
        int K = 100;
        int width = 128;//56;        // TODO
    
    
    public:
        dnnType *rt_out[4];

        float inp_height = 512;//224;//512;
        float inp_width = 512;//224;//512;
    
        int classes = 80;
        int num = 0;
        int n_masks = 0;
        float thresh = 0.3;
        cv::Scalar colors[256];

        // this is filled with results
        std::vector<tk::dnn::box> detected;
        // draw
        std::vector<std::string> coco_class_name;

        // keep track of inference times (ms)
        std::vector<double> stats;
        
        CenternetDetection() {}

        virtual ~CenternetDetection() {}

        /**
         * Method used for inizialize the class
         * 
         * @return Success of the initialization
         */
        bool init(std::string tensor_path);
        void testdog();
        cv::Mat draw(cv::Mat &frame);
        void update(cv::Mat &frame);

};

}}
