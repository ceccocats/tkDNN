#ifndef DEPTHNN_H
#define DEPTHNN_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>    
#ifdef __linux__
#include <unistd.h>
#endif 

#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkDNN/utils.h"
#include "tkDNN/tkdnn.h"

#include "NetworkViz.h"


namespace tk { namespace dnn {

class DepthNN {

    public:
        tk::dnn::NetworkRT *netRT = nullptr;
        dnnType *input_h;
        dnnType *input_d;
        float* depth_h;

        int output_w;
        int output_h;

        int nBatches = 1;

        cv::Mat bgr[3];
        cv::Mat imagePreproc;

        std::vector<double> stats; /*keeps track of inference times (ms)*/
        std::vector<std::vector<float>> depths;
        std::vector<cv::Mat> depthMats;

        DepthNN() {};
        ~DepthNN(){};

        /**
         * Method used to initialize the class, allocate memory and compute 
         * needed data.
         * 
         * @param tensor_path path to the rt file of the NN.
         * @param n_batches maximum number of batches to use in inference
         * @return true if everything is correct, false otherwise.
         */
        void init(const std::string& tensor_path, const int n_batches=1){
            //create net
            
            std::cout<<(tensor_path).c_str()<<"\n";
            nBatches = n_batches;
            netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str());
            
            //allocate memory for NN input
            checkCuda(cudaMallocHost(&input_h, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));
            checkCuda(cudaMalloc(&input_d, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));

            //allocate memory for NN output
            depthMats.resize(nBatches);
            depths.resize(nBatches);
            for(int i=0; i< depths.size();++i)
                depths[i].resize(netRT->buffersDIM[1].tot());

            depth_h = (float *)malloc(netRT->buffersDIM[1].tot() * sizeof(float));

            output_h = netRT->buffersDIM[1].h;
            output_w = netRT->buffersDIM[1].w;
            
        }


        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param frame original frame to adapt for inference.
         * @param bi batch index
         */
        void preprocess(cv::Mat &frame, const int bi=0) {
            //resize image, remove mean, divide by std
            cv::Mat frame_nomean;
            resize(frame, frame, cv::Size(netRT->input_dim.w, netRT->input_dim.h));
            frame.convertTo(frame_nomean, CV_32FC3);
            frame_nomean.convertTo(imagePreproc, CV_32FC3, 1 / 255.0, 0);

            //copy image into tensor and copy it into GPU
            cv::split(imagePreproc, bgr);
            for (int i = 0; i < netRT->input_dim.c; i++){
                int idx = i * imagePreproc.rows * imagePreproc.cols;
                int ch = netRT->input_dim.c-1 -i;
                memcpy((void *)&input_h[idx + netRT->input_dim.tot()*bi], (void *)bgr[ch].data, imagePreproc.rows * imagePreproc.cols * sizeof(dnnType));
            }
            checkCuda(cudaMemcpyAsync(input_d+ netRT->input_dim.tot()*bi, input_h + netRT->input_dim.tot()*bi, netRT->input_dim.tot() * sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));
        }

        /**
         * This method postprocess the output of the NN to obtain the correct 
         * boundig boxes. 
         * 
         * @param bi batch index
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation
         */
        void postprocess(const int bi=0) {

            dnnType *rt_out[1];
            rt_out[0] = (dnnType *)netRT->buffersRT[1]+ netRT->buffersDIM[1].tot()*bi;
            checkCuda(cudaMemcpy(depth_h, rt_out[0], netRT->buffersDIM[1].tot()* sizeof(float), cudaMemcpyDeviceToHost));
            memcpy(&depths[bi][0], &depth_h[0], netRT->buffersDIM[1].tot()* sizeof(float));

            // cv::Mat d(netRT->buffersDIM[1].h, netRT->buffersDIM[1].w, CV_8UC1, depth_h);
            // depthMats[bi] = d.clone();

            cv::Mat depth_mat = vizData2Mat(rt_out[0], netRT->buffersDIM[1], netRT->buffersDIM[1].h, netRT->buffersDIM[1].w);
            // cv::Mat depth_mat = vizData2Mat((dnnType *)netRT->buffersRT[0], netRT->buffersDIM[0], netRT->buffersDIM[0].h, netRT->buffersDIM[0].w);
            depthMats[bi] = depth_mat.clone();

        }
        
        /**
         * This method performs the inference of the NN.
         * 
         * @param frames frames to build the embedding from.
         * @param cur_batches number of batches to use in inference
         */
        void update(std::vector<cv::Mat>& frames, const int cur_batches=1){
            if(cur_batches > nBatches)
                FatalError("A batch size greater than nBatches cannot be used");

            if(TKDNN_VERBOSE) printCenteredTitle(" TENSORRT feature extraction ", '=', 30); 
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi){
                    if(!frames[bi].data)
                        FatalError("No image data feed to extract features");
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
         * Method to draw the result.
         * 
         */
        void draw() {        }

};


}}

#endif /* DEPTHNN_H*/
