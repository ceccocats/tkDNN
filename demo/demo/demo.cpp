#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
//#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, sig_handler);

#ifdef __linux__
    std::string config_file = "../demo/demoConfig.yaml";
#elif _WIN32
    std::string config_file = "..\\..\\..\\demo\\demoConfig.yaml";
#endif

    if(argc > 1){
        config_file = argv[1];
    }

    YAML::Node conf = YAMLloadConf(config_file);
    if(!conf){
        FatalError("Problem with config file");
    }


    std::string net = YAMLgetConf<std::string>(conf,"net","yolo4tiny_fp32.rt");
    if(!fileExist(net.c_str())) {
        FatalError("The given network does not exist. Create the rt first.");
    }

#ifdef __linux__
    std::string input = YAMLgetConf<std::string>(conf, "input", "../demo/yolo_test.mp4");
    std::string cfgPath = YAMLgetConf<std::string>(conf,"cfg_input", "../tests/darknet/cfg/yolo4tiny.cfg");
    std::string namePath = YAMLgetConf<std::string>(conf,"name_input","../tests/darknet/names/coco.names");
#elif _WIN32
    std::string input = YAMLgetConf(conf, "win_input", "..\\..\\..\\demo\\yolo_test.mp4");
    std::string cfgPath = YAMLgetConf(conf,"cfg_win_input","..\\..\\..\\tests\\darknet\\cfg\\yolo4tiny.cfg");
    std::string namePath = YAMLgetConf(conf,"name_win_input","..\\..\\..\\tests\\darknet\\names\\coco.names");
#endif
    if(!fileExist(input.c_str()))
    FatalError("The given input video does not exist.");

    char ntype          = YAMLgetConf<char>(conf, "ntype", 'y');
    int n_classes       = YAMLgetConf<int>(conf, "n_classes", 80);
    int n_batch         = YAMLgetConf<int>(conf, "n_batch", 1);
    if(n_batch < 1 || n_batch > 64)
    FatalError("Batch dim not supported");
    float conf_thresh   = YAMLgetConf<float>(conf, "conf_thresh", 0.3);
    bool show           = YAMLgetConf<bool>(conf, "show", true);
    bool save           = YAMLgetConf<bool>(conf, "save", false);

    std::cout   <<"Net settings - net: "<< net
                <<", ntype: "<< ntype
                <<", n_classes: "<< n_classes
                <<", n_batch: "<< n_batch
                <<", conf_thresh: "<< conf_thresh<<"\n"; 
    std::cout   <<"Demo settings - input: "<< input
                <<", show: "<< show
                <<", save: "<< save<<"\n\n"; 

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;

    tk::dnn::DetectionNN *detNN;  

    switch(ntype)
    {
        case 'y':
            detNN = &yolo;
            break;
        case 'c':
            detNN = &cnet;
            break;
        case 'm':
            detNN = &mbnet;
            n_classes++;
            break;
        default:
        FatalError("Network type not allowed (3rd parameter)\n");
    }

    if(ntype == 'c' || ntype == 'm'){
        cfgPath = "";
        namePath = "";
    }

    detNN->init(net,cfgPath,namePath,n_classes,n_batch,conf_thresh);

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::VideoWriter resultVideo;
    if(save) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    while(gRun) {
        batch_dnn_input.clear();
        batch_frame.clear();
        
        for(int bi=0; bi< n_batch; ++bi){
            cap >> frame; 
            if(!frame.data) 
                break;
            
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        } 
        if(!frame.data) 
            break;
    
        //inference
        detNN->update(batch_dnn_input, n_batch);
        detNN->draw(batch_frame);

        if(show){
            for(int bi=0; bi< n_batch; ++bi){
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
            }
        }
        if(n_batch == 1 && save)
            resultVideo << frame;
    }

    std::cout<<"detection end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for(int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;   
    

    return 0;
}

