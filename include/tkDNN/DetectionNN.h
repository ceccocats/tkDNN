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

#include "tkdnn.h"

//#define OPENCV_CUDACONTRIB //if OPENCV has been compiled with CUDA and contrib.

#ifdef OPENCV_CUDACONTRIB
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif


namespace tk { namespace dnn {

class DetectionNN {

    protected:
        tk::dnn::NetworkRT *netRT = nullptr;
        dnnType *input_d;

        cv::Size originalSize;

        cv::Scalar colors[256];

#ifdef OPENCV_CUDACONTRIB
        cv::cuda::GpuMat bgr[3];
        cv::cuda::GpuMat imagePreproc;
#else
        cv::Mat bgr[3];
        cv::Mat imagePreproc;
        dnnType *input;
#endif

        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param frame original frame to adapt for inference.
         */
        virtual void preprocess(cv::Mat &frame) = 0;

        /**
         * This method postprocess the output of the NN to obtain the correct 
         * boundig boxes. 
         * 
         */
        virtual void postprocess() = 0;

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
         * @param tensor_path path to the rt file og the NN.
         * @param n_classes number of classes for the given dataset.
         * @return true if everything is correct, false otherwise.
         */
        virtual bool init(const std::string& tensor_path, const int n_classes=80) = 0;
        
        /**
         * This method performs the whole detection of the NN.
         * 
         * @param frame frame to run detection on.
         * @param save_times if set to true, preprocess, inference and postprocess times 
         *        are saved on a csv file, otherwise not.
         * @param times pointer to the output stream where to write times
         */
        void update(cv::Mat &frame, bool save_times=false, std::ofstream *times=nullptr){
            if(!frame.data)
                FatalError("No image data feed to detection");

            if(save_times && times==nullptr)
                FatalError("save_times set to true, but no valid ofstream given");

            originalSize = frame.size();
            printCenteredTitle(" TENSORRT detection ", '=', 30); 
            {
                TIMER_START
                preprocess(frame);
                TIMER_STOP
                if(save_times) *times<<t_ns<<";";
            }

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            {
                dim.print();
                TIMER_START
                netRT->infer(dim, input_d);
                TIMER_STOP
                dim.print();
                stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<";";
            }

            {
                TIMER_START
                postprocess();
                TIMER_STOP
                if(save_times) *times<<t_ns<<"\n";
            }
        }      

        /**
         * Method to draw boundixg boxes and labels on a frame.
         * 
         * @param frame orginal frame to draw bounding box on.
         * @return frame with boundig boxes.
         */
        cv::Mat draw(cv::Mat &frame) {
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
