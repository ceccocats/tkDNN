#ifndef DETECTIONNN_H
#define DETECTIONNN_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>    
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


#include "tkdnn.h"

#define OPENCV_CUDA //if OPENCV has been compiled with CUDA and contrib.

namespace tk { namespace dnn {

enum networkType_t{
    NETWORK_YOLO3,
    NETWORK_MOBILENETSSDLITE,
    NETWORK_CENTERNET
};

class DetectionNN {

    protected:
        tk::dnn::NetworkRT *netRT = nullptr;
        dnnType *input_d;

        cv::Size originalSize;

        cv::Scalar colors[256];

#ifdef OPENCV_CUDA
        cv::cuda::GpuMat bgr[3];
        cv::cuda::GpuMat imagePreproc;
#else
        cv::Mat bgr[3];
        cv::Mat imagePreproc;
        dnnType *input;
#endif

    public:
        int classes = 0;
        float confThreshold = 0.3; /*threshold on the confidence of the boxes*/

        std::vector<tk::dnn::box> detected; /*bounding boxes in output*/
        std::vector<double> stats; /*keeps track of inference times (ms)*/
        std::vector<std::string> classesNames;

        DetectionNN() {};
        ~DetectionNN(){};

        /**
         * Method used to inialize the class, allocate memory and compute 
         * needed data.
         * 
         * @param path to the rt file og the NN.
         * @return true if everything is correct, false otherwise.
         */
        virtual bool init(const std::string& tensor_path, const int n_classes=80) = 0;
        
        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param original frame to adapt for inference.
         */
        virtual void preprocess(cv::Mat &frame) = 0;

        /**
         * This method performs the whole detection of the NN.
         * 
         * @param frame to run detection on.
         */
        virtual void update(cv::Mat &frame) = 0;

        /**
         * This method postprocess the output of the NN to obtain the correct 
         * boundig boxes. 
         * 
         * @param outputs of the inference
         * @param number of outputs of the inference
         */
        virtual void postprocess(dnnType **rt_out, const int n_out) = 0;

        /**
         * Method to draw boundixg boxes and labels on a frame.
         * 
         * @param orginal frame to draw bounding box on.
         * @return frame with boundig boxes.
         */
        cv::Mat draw(cv::Mat &frame) 
        {
            tk::dnn::box b;
            int x0, w, x1, y0, h, y1;
            int objClass;
            std::string det_class;

            int baseline = 0;
            float font_scale = 0.5;
            int thickness = 2;   
            // draw dets
            for(int i=0; i<detected.size(); i++) {
                b           = detected[i];
                x0   		= b.x;
                x1   		= b.x + b.w;
                y0   		= b.y;
                y1   		= b.y + b.h;
                det_class 	= classesNames[b.cl];

                // draw rectangle
                cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2); 

                // draw label
                cv::Size text_size = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
                cv::rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), colors[b.cl], -1);                      
                cv::putText(frame, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
            }
            return frame;
        }

};

}}

#endif /* DETECTIONNN_H*/
