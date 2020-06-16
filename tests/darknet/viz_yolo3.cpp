#include<iostream>
#include<vector>
#include <opencv2/highgui/highgui.hpp>

#include "tkdnn.h"
#include "test.h"
#include "DarknetParser.h"
#include "NetworkViz.h"

int main() {
    std::string bin_path  = "yolo3";
    std::vector<std::string> input_bins = { 
        bin_path + "/layers/input.bin"
    };
    std::string wgs_path  = bin_path + "/layers";
    std::string cfg_path  = std::string(TKDNN_PATH) + "/tests/darknet/cfg/yolo3.cfg";
    std::string name_path = std::string(TKDNN_PATH) + "/tests/darknet/names/coco.names";
    downloadWeightsifDoNotExist(input_bins[0], bin_path, "https://cloud.hipert.unimore.it/s/jPXmHyptpLoNdNR/download");

    // parse darknet network
    tk::dnn::Network *net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);
    net->print();

    // Load input and infer
    dnnType *input_d;
    dnnType *input_h;
    readBinaryFile(input_bins[0], net->input_dim.tot(), &input_h, &input_d);
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

    delete [] input_h;
    checkCuda(cudaFree(input_d));
    net->releaseLayers();
    delete net;
    return 0;
}

 