#include<iostream>
#include<vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tkdnn.h"
#include "test.h"
#include "DarknetParser.h"
#include "NetworkViz.h"

int main(int argc, char *argv[]) {
    if(argc <2)
        FatalError("you must provide an input image");
    std::string input_image = argv[1];
    std::string bin_path  = "yolo3";
    std::string wgs_path  = bin_path + "/layers";
    std::string cfg_path  = std::string(TKDNN_PATH) + "/tests/darknet/cfg/yolo3.cfg";
    std::string name_path = std::string(TKDNN_PATH) + "/tests/darknet/names/coco.names";
    downloadWeightsifDoNotExist(wgs_path, bin_path, "https://cloud.hipert.unimore.it/s/jPXmHyptpLoNdNR/download");

    // parse darknet network
    tk::dnn::Network *net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);
    net->print();

    // input data
    dnnType *input_d;
    checkCuda( cudaMalloc(&input_d, sizeof(dnnType)*net->input_dim.tot()));

    // load image
    cv::Mat frame, frameFloat;
    frame = cv::imread(input_image);
    cv::resize(frame, frame, cv::Size(net->input_dim.w, net->input_dim.h));
    frame.convertTo(frameFloat, CV_32FC3, 1/255.0); 

    //split channels
    cv::Mat bgr[3];
    cv::split(frameFloat,bgr);//split source

    //write channels
    for(int i=0; i<net->input_dim.c; i++) {
        int idx = i*frameFloat.rows*frameFloat.cols;
        int ch = net->input_dim.c-1 -i;
        checkCuda( cudaMemcpy(input_d + idx, (void*)bgr[ch].data, frameFloat.rows*frameFloat.cols*sizeof(dnnType), cudaMemcpyHostToDevice));     
    }
    
    tk::dnn::dataDim_t dim =  net->input_dim;     
    dim.print();
    std::cout<<"infer\n";
    net->infer(dim, input_d);    

    // output directory
    std::string output_viz = "viz/";
    system( (std::string("mkdir -p ") + output_viz).c_str() );

    for(int i=0; i<net->num_layers; i++) {
        std::string output_png = output_viz + "/layer" + std::to_string(i) + ".png";
        std::cout<<"saving "<<output_png<<"\n";
        cv::Mat viz = vizLayer2Mat(net, i);
        cv::imwrite(output_png, viz);
        //cv::imshow("layer", viz);
        //cv::waitKey(0);
    }

    checkCuda(cudaFree(input_d));
    net->releaseLayers();
    delete net;
    return 0;
}

 