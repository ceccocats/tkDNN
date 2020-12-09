#ifndef DETECTIONNN3D_H
#define DETECTIONNN3D_H

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

// #define OPENCV_CUDACONTRIB //if OPENCV has been compiled with CUDA and contrib.

#ifdef OPENCV_CUDACONTRIB
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif


namespace tk { namespace dnn {

class DetectionNN3D {

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
        std::vector<double> pre_stats, stats, post_stats, visual_stats; /*keeps track of inference times (ms)*/
        std::vector<std::string> classesNames;

        DetectionNN3D() {};
        ~DetectionNN3D(){};

        /**
         * Method used to inialize the class, allocate memory and compute 
         * needed data.
         * 
         * @param tensor_path path to the rt file og the NN.
         * @param n_classes number of classes for the given dataset.
         * @return true if everything is correct, false otherwise.
         */
        virtual bool init(const std::string& tensor_path, const int n_classes=3) = 0;

        /**
         * Method to draw boundixg boxes and labels on a frame.
         * 
         * @param frame orginal frame to draw bounding box on.
         * @return frame with boundig boxes.
         */
        virtual cv::Mat draw(cv::Mat &frame){};

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
                TKDNN_TSTART
                preprocess(frame);
                TKDNN_TSTOP
                pre_stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<";";
            }

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            {
                dim.print();
                TKDNN_TSTART
                netRT->infer(dim, input_d);
                TKDNN_TSTOP
                dim.print();
                stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<";";
            }

            {
                TKDNN_TSTART
                postprocess();
                TKDNN_TSTOP
                post_stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<"\n";
            }
        }              
};

}}

#endif /* DETECTIONNN3D_H*/
