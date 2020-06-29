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
        std::vector<cv::Mat> masks;
        cv::Mat bgr[3];
        dnnType *input;
        dnnType *input_d;
        float* confidences_h;

        float * tmpInputData_d;
        float *tmpOutData_d;
        float *tmpOutData_h;

        float *mean_d, *stddev_d;

        cublasHandle_t cublasHandle;

        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param frame original frame to adapt for inference.
         * @param bi batch index
         */
        void preprocess(cv::Mat &frame, const int bi=0) {
            frame.convertTo(frame, CV_32FC3, 1 / 255.0, 0);
            int H = frame.rows;
            int W = frame.cols;
            cv::Mat frame_cropped;
            cv::Mat mask(frame.size(), CV_8UC3, cv::Scalar(255,255,255));
            
            if(H != W){
                if(H < W){
                    int top = (W - H)/2;
                    int bottom = W - top - H;
                    cv::copyMakeBorder(frame, frame_cropped, top, bottom, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
                    cv::copyMakeBorder(mask, mask, top, bottom, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
                }
                else{
                    int left = (H - W)/2;
                    int right = H - left - W;
                    cv::copyMakeBorder(frame, frame_cropped, 0, 0, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
                    cv::copyMakeBorder(mask, mask, 0, 0, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
                }
            }

            tk::dnn::dataDim_t idim = netRT->input_dim;

            resize(frame_cropped, frame_cropped, cv::Size(idim.w, idim.h));
            resize(mask, mask, cv::Size(idim.w, idim.h));
            masks[bi] = mask;
            
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
        void postprocess(const int bi=0) {
            dnnType *rt_out = (dnnType *)netRT->buffersRT[1]+ netRT->buffersDIM[1].tot()*bi;

            dataDim_t odim = netRT->output_dim;
            
            matrixTranspose(cublasHandle, rt_out, tmpInputData_d, odim.c, odim.w*odim.h);
            maxElem(tmpInputData_d, tmpOutData_d, odim.c, odim.h, odim.w);
            checkCuda(cudaMemcpy(tmpOutData_h, tmpOutData_d, odim.w*odim.h * sizeof(float), cudaMemcpyDeviceToHost));

            dataDim_t vdim = odim;
            vdim.c = 1;
            segmented[bi] = vizData2Mat(tmpOutData_h, vdim, 1024, 0, 18);
        };

    public:
        int classes = 0;
        std::vector<double> stats; /*keeps track of inference times (ms)*/
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
            masks.resize(nBatches);

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
        void update(std::vector<cv::Mat>& frames, const int cur_batches=1){
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
                    postprocess(bi);
                TKDNN_TSTOP
            }
        }      

        /**
         * Method to draw boundixg boxes and labels on a frame.
         */
        cv::Mat draw(const int cur_batches=1) {
            for(int i=0; i<cur_batches; ++i){

                cv::bitwise_and(segmented[i], masks[i], segmented[i]);

                cv::imshow("segmented", segmented[i]);
                cv::waitKey(1);
            }
            return segmented[0];
        }

};

}}

#endif /* SEGMENTATIONNN_H*/
