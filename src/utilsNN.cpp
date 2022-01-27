#include "kernels.h"
#include "utilsNN.h"


void resizeAndSplit(cv::Mat& frame, uint8_t** frame_d, int& frame_size, dnnType *input_d, tk::dnn::NetworkRT *netRT, const int bi, bool BGR){
    int new_frame_size = sizeof(uint8_t) * frame.cols * frame.rows * frame.channels();
    if(*frame_d == nullptr){
        frame_size = new_frame_size;
        checkCuda(cudaMalloc(frame_d, frame_size));
    }
    else{
        if(new_frame_size > frame_size ){
            frame_size = new_frame_size;
            checkCuda(cudaFree(frame_d));
            checkCuda(cudaMalloc(frame_d, frame_size));
        }
    }
    
    checkCuda(cudaMemcpyAsync(*frame_d, frame.data, frame_size, cudaMemcpyHostToDevice, netRT->stream));
    if(BGR){
        interleavedRGBToPlanarBGR(*frame_d,
                                input_d + netRT->input_dim.tot() * bi, 
                                frame.cols, 
                                frame.rows, 
                                frame.channels(),
                                netRT->input_dim.w,
                                netRT->input_dim.h
                            );
    }
    else{
        interleavedToPlanar(*frame_d,
                                input_d + netRT->input_dim.tot() * bi, 
                                frame.cols, 
                                frame.rows, 
                                frame.channels(),
                                netRT->input_dim.w,
                                netRT->input_dim.h
                            );

    }
}