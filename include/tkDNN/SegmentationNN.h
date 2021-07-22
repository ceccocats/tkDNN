#ifndef SEGMENTATIONNN_H
#define SEGMENTATIONNN_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>    
#include <unistd.h>
#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>

#include "tkdnn.h"
#include "NetworkViz.h"
#include "kernelsThrust.h"

namespace tk { namespace dnn {

class SegmentationNN {

    protected:
        tk::dnn::NetworkRT *netRT = nullptr;
        int nBatches = 1;

        std::vector<cv::Size> originalSize;
        cv::Mat bgr[3];
        dnnType *input;
        dnnType *input_d;
        float* confidences_h;

        float * tmpInputData_d;
        float *tmpOutData_d;
        float *tmpOutData_h;

        float *mean_d, *stddev_d;

        cublasHandle_t cublasHandle;

        void computeBorders(const int or_width, const int or_height, int& top, int& bottom, int& left, int&right){
            top = 0;
            bottom = 0;
            left = 0;
            right = 0;

            if(or_height != or_width){
                if(or_height < or_width){
                    top = (or_width - or_height)/2;
                    bottom = or_width - top - or_height;
                }
                else{
                    left = (or_height - or_width)/2;
                    right = or_height - left - or_width;
                }
            }
        }

        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param frame original frame to adapt for inference.
         * @param bi batch index
         */
        void preprocess(cv::Mat &frame, const int bi=0) {
            originalSize[bi] = frame.size();

            frame.convertTo(frame, CV_32FC3, 1 / 255.0, 0);
            int H = frame.rows;
            int W = frame.cols;
            cv::Mat frame_cropped;

            int top, bottom, left, right;
            computeBorders(W, H, top, bottom, left, right);
            cv::copyMakeBorder(frame, frame_cropped, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

            tk::dnn::dataDim_t idim = netRT->input_dim;

            resize(frame_cropped, frame_cropped, cv::Size(idim.w, idim.h));
            
            cv::split(frame_cropped, bgr);
            for (int i = 0; i < idim.c; i++){
                int idx = i * frame_cropped.rows * frame_cropped.cols;
                int ch = idim.c-1 -i;
                memcpy((void *)&input[idx + idim.tot()*bi], (void *)bgr[ch].data, frame_cropped.rows * frame_cropped.cols * sizeof(dnnType));
            }

            checkCuda(cudaMemcpyAsync(input_d+ idim.tot()*bi, input + idim.tot()*bi, idim.tot() * sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));

            normalize(input_d + idim.tot()*bi, idim.c, idim.h, idim.w, mean_d, stddev_d);
        }        

        /**
         * This method postprocess the output of the NN to obtain the correct 
         * boundig boxes. 
         * 
         * @param bi batch index
         */
        void postprocess(const int bi=0, bool appy_colormap = true) {
            dnnType *rt_out = (dnnType *)netRT->buffersRT[1]+ netRT->buffersDIM[1].tot()*bi;

            dataDim_t odim = netRT->output_dim;
            
            matrixTranspose(cublasHandle, rt_out, tmpInputData_d, odim.c, odim.w*odim.h);
            maxElem(tmpInputData_d, tmpOutData_d, odim.c, odim.h, odim.w);
            checkCuda(cudaMemcpy(tmpOutData_h, tmpOutData_d, odim.w*odim.h * sizeof(float), cudaMemcpyDeviceToHost));

            dataDim_t vdim = odim;
            vdim.c = 1;

            cv::Mat colored;

            if(appy_colormap)
                colored = vizData2Mat(tmpOutData_h, vdim, netRT->input_dim.h, netRT->input_dim.w, 0, classes, classes);
            else{
                cv::Mat colored_fp32 (cv::Size(odim.w, odim.h),CV_32FC1, tmpOutData_h);
                colored_fp32.convertTo(colored, CV_8UC1);
            }
            
            int max_dim = (originalSize[bi].width > originalSize[bi].height) ? originalSize[bi].width : originalSize[bi].height;
            resize(colored, colored, cv::Size(max_dim, max_dim));
            int top, bottom, left, right;
            computeBorders(originalSize[bi].width, originalSize[bi].height, top, bottom, left, right);
            cv::Rect roi(left,top,originalSize[bi].width, originalSize[bi].height);
            cv::Mat or_size (colored, roi);
            segmented[bi] = or_size;
        };

    public:
        int classes = 0;
        std::vector<double> stats; /*keeps track of inference times (ms)*/
        std::vector<double> stats_pre; 
        std::vector<double> stats_post; 
        std::vector<std::string> classesNames;
        std::vector<cv::Mat> segmented;

        SegmentationNN() {
            checkERROR( cublasCreate(&cublasHandle) );
        };
        ~SegmentationNN(){
            checkERROR( cublasDestroy(cublasHandle) );
        };

        /**
         * Method used to inialize the class, allocate memory and compute 
         * needed data.
         * 
         * @param tensor_path path to the rt file og the NN.
         * @param n_classes number of classes for the given dataset.
         * @param n_batches maximum number of batches to use in inference
         * @return true if everything is correct, false otherwise.
         */
        bool init(const std::string& tensor_path, const int n_classes=19, const int n_batches=1){
            std::cout<<(tensor_path).c_str()<<"\n";
            if(!fileExist(tensor_path.c_str()))
                FatalError("This file do not exists" + tensor_path );

            netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str());
            classes = n_classes;
            nBatches = n_batches;

            checkCuda(cudaMallocHost(&input, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));
            checkCuda(cudaMalloc(&input_d, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));

            dataDim_t odim = netRT->output_dim;

            checkCuda(cudaMallocHost(&confidences_h, sizeof(float) * odim.tot()));
            checkCuda(cudaMalloc(&tmpInputData_d, sizeof(float) * odim.tot()));
            checkCuda(cudaMalloc(&tmpOutData_d, sizeof(float) * odim.w*odim.h));
            checkCuda(cudaMallocHost(&tmpOutData_h, sizeof(float) * odim.w*odim.h));

            segmented.resize(nBatches);
            originalSize.resize(nBatches);

            std::vector<float> mean = {0.485, 0.456, 0.406};
            std::vector<float> stddev = {0.229, 0.224, 0.225};

            checkCuda(cudaMalloc(&mean_d, sizeof(float) * mean.size()));
            checkCuda(cudaMalloc(&stddev_d, sizeof(float) * stddev.size()));

            checkCuda(cudaMemcpyAsync(mean_d, mean.data(), mean.size() * sizeof(float), cudaMemcpyHostToDevice, netRT->stream));
            checkCuda(cudaMemcpyAsync(stddev_d, stddev.data(), stddev.size() * sizeof(float), cudaMemcpyHostToDevice, netRT->stream));

        }
        
        /**
         * This method performs the whole detection of the NN.
         * 
         * @param frames frames to run detection on.
         * @param cur_batches number of batches to use in inference
         * @param save_times if set to true, preprocess, inference and postprocess times 
         *        are saved on a csv file, otherwise not.
         * @param times pointer to the output stream where to write times
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation
         */
        void update(std::vector<cv::Mat>& frames, const int cur_batches=1, bool apply_colormap=true){
            if(cur_batches > nBatches)
                FatalError("A batch size greater than nBatches cannot be used");

            originalSize.clear();
            if(TKDNN_VERBOSE) printCenteredTitle(" TENSORRT detection ", '=', 30); 
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi){
                    if(!frames[bi].data)
                        FatalError("No image data feed to detection");
                    originalSize.push_back(frames[bi].size());
                    preprocess(frames[bi], bi);    
                }
                TKDNN_TSTOP
                stats_pre.push_back(t_ns);
            }

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            dim.n = cur_batches;
            {
                if(TKDNN_VERBOSE) dim.print();
                TKDNN_TSTART
                netRT->infer(dim, input_d);
                TKDNN_TSTOP
                if(TKDNN_VERBOSE) dim.print();
                stats.push_back(t_ns);
            }

            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi)
                    postprocess(bi, apply_colormap);
                TKDNN_TSTOP
                stats_post.push_back(t_ns);
            }
        }     

        void updateOriginal(cv::Mat frame, bool apply_colormap=true){

            std::vector<cv::Mat> splitted_frames;
            int H, W, net_H, net_W;
            int top = 0, bottom = 0, left = 0, right = 0;
            std::vector<std::pair<int,int>> pos;

            {
                TKDNN_TSTART
                cv::Size original_size =  frame.size();

                frame.convertTo(frame, CV_32FC3, 1 / 255.0, 0);
                H = frame.rows;
                W = frame.cols;
                net_H = netRT->input_dim.h; 
                net_W = netRT->input_dim.w;

                cv::Mat frame_cropped;
                
                if( H <= net_H && W <= net_W ){ // smaller size wrt network
                    top = (net_H - H)/2;
                    bottom = net_H - H - top ;
                    left = (net_W - W)/2;
                    right = net_W - W - left ;
                    cv::copyMakeBorder(frame, frame_cropped, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
                    splitted_frames.push_back(frame_cropped);
                }
                else{ //bigger size wrt network

                    
                    if(H < net_H  ||  W < net_W){
                        if(H < net_H){
                            top = (net_H - H)/2;
                            bottom = net_H - H - top ;
                        }
                        else{
                            left = (net_W - W)/2;
                            right = net_W - W - left ;
                        }
                        cv::copyMakeBorder(frame, frame_cropped, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
                    }

                    for(int x=0; x+net_W<=W ;){
                        for(int y=0; y+net_H <=H ; ){
                            cv::Rect roi(x, y, net_W, net_H);
                            cv::Mat image_roi = frame(roi);
                            splitted_frames.push_back(image_roi);
                            pos.push_back(std::make_pair(x,y));
                            
                            y += net_H;
                            if(y == H)
                                break;
                            if(y + net_H > H) y = H - net_H;
                        }
                        x += net_W;
                        if(x == W)
                            break;
                        if(x + net_W > W) x = W - net_W;
                    }
                }

                tk::dnn::dataDim_t idim = netRT->input_dim;

                if(splitted_frames.size()> nBatches)
                    FatalError(std::to_string(splitted_frames.size()) + " min batches required");

                for(int bi=0; bi<splitted_frames.size();++bi){
                    cv::split(splitted_frames[bi], bgr);
                    for (int i = 0; i < idim.c; i++){
                        int idx = i * splitted_frames[bi].rows * splitted_frames[bi].cols;
                        int ch = idim.c-1 -i;
                        memcpy((void *)&input[idx + idim.tot()*bi], (void *)bgr[ch].data, splitted_frames[bi].rows * splitted_frames[bi].cols * sizeof(dnnType));
                    }

                    checkCuda(cudaMemcpyAsync(input_d+ idim.tot()*bi, input + idim.tot()*bi, idim.tot() * sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));
                    normalize(input_d + idim.tot()*bi, idim.c, idim.h, idim.w, mean_d, stddev_d);
                }
                TKDNN_TSTOP
                stats_pre.push_back(t_ns);
            }

            tk::dnn::dataDim_t dim = netRT->input_dim;
            dim.n = splitted_frames.size();
            {
                if(TKDNN_VERBOSE) dim.print();
                TKDNN_TSTART
                netRT->infer(dim, input_d);
                TKDNN_TSTOP
                if(TKDNN_VERBOSE) dim.print();
                stats.push_back(t_ns);
            }

            dataDim_t odim = netRT->output_dim;

            std::vector<cv::Mat> out_img;

            {
                TKDNN_TSTART

                for(int bi=0; bi<splitted_frames.size();++bi){

                    dnnType *rt_out = (dnnType *)netRT->buffersRT[1]+ netRT->buffersDIM[1].tot()*bi;               
                    
                    matrixTranspose(cublasHandle, rt_out, tmpInputData_d, odim.c, odim.w*odim.h);
                    maxElem(tmpInputData_d, tmpOutData_d, odim.c, odim.h, odim.w);
                    checkCuda(cudaMemcpy(tmpOutData_h, tmpOutData_d, odim.w*odim.h * sizeof(float), cudaMemcpyDeviceToHost));

                    dataDim_t vdim = odim;
                    vdim.c = 1;

                    cv::Mat colored;

                    if(apply_colormap)
                        colored = vizData2Mat(tmpOutData_h, vdim, netRT->input_dim.h, netRT->input_dim.w, 0, classes, classes);
                    else{
                        cv::Mat colored_fp32 (cv::Size(odim.w, odim.h),CV_32FC1, tmpOutData_h);
                        colored_fp32.convertTo(colored, CV_8UC1);
                    }
                    out_img.push_back(colored);
                }


                cv::Mat seg(frame.size(), out_img[0].type());
                if(out_img.size() == 1)
                {
                    cv::Rect roi(left, top, W, H);
                    seg = out_img[0](roi);
                }
                else{
                    int bi=0;

                    if(top == 0 && left == 0){

                        for(int i=0; i<out_img.size(); ++i){
                            cv::Mat roi_collage = seg(cv::Rect( pos[i].first ,pos[i].second,out_img[i].cols,out_img[i].rows));
                            out_img[i].copyTo(roi_collage);
                        }
                    }
                    else{
                        FatalError("Not handled case")
                    }
                }
                segmented[0] = seg;
                
                TKDNN_TSTOP
                stats_post.push_back(t_ns);
            }
        } 

        /**
         * Method to draw boundixg boxes and labels on a frame.
         */
        cv::Mat draw(const int cur_batches=1) {
            for(int i=0; i<cur_batches; ++i){

                cv::imshow("segmented", segmented[i]);
                cv::resizeWindow("segmented", cv::Size(512,288));
                cv::waitKey(1);
            }
            return segmented[0];
        }

};

}}

#endif /* SEGMENTATIONNN_H*/
