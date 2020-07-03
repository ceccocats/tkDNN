#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>

#include "SegmentationNN.h"

bool gRun;
bool SAVE_RESULT = true;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

void writePred(const std::string& images_names, const std::string& gt_folder, const std::string& out_folder, tk::dnn::SegmentationNN& segNN){
    std::ifstream all_gt(images_names);
    std::string filename;
    cv::Mat frame;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    for (; std::getline(all_gt, filename); ) {
        std::cout<<filename<<std::endl;
        frame = cv::imread(gt_folder + filename);
        batch_dnn_input.clear();
        batch_frame.clear();
        batch_frame.push_back(frame);
        batch_dnn_input.push_back(frame.clone());
        segNN.update(batch_dnn_input, 1, false);
        cv::imwrite(out_folder + filename, segNN.segmented[0]);
    }
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);


    std::string net = "shelfnet_fp32.rt";
    if(argc > 1)
        net = argv[1]; 
    std::string input = "../../ShelfNet/ShelfNet18_realtime/data/leftImg8bit/test/modena/000302.png";
    if(argc > 2)
        input = argv[2]; 
    int n_batch = 1;
    if(argc > 3)
        n_batch = atoi(argv[3]); 
    int n_classes = 19;
    if(argc > 4)
        n_classes = atoi(argv[4]); 
    bool show = false;
    if(argc > 5)
        show = atoi(argv[5]); 
    bool write_pred = false;
    if(argc > 6)
        write_pred = atoi(argv[6]); 
    

    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    tk::dnn::SegmentationNN segNN;
    segNN.init(net, n_classes, n_batch);

    if(write_pred){
        std::string gt_folder = "../demo/CityScapes_val/images/";
        std::string images_names = "../demo/CityScapes_val/all_images.txt";
        std::string out_folder = "seg/";

        writePred(images_names, gt_folder, out_folder, segNN);
        return 0;
    }

    if(!show)
        SAVE_RESULT = true;

    gRun = true;

    cv::VideoCapture cap(input);
    if(!cap.isOpened())
        gRun = false; 
    else
        std::cout<<"camera started\n";

    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(1024, 1024));
    }

    cv::Mat frame;
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    int height = 0, width = 0;

    while(gRun) {
        batch_dnn_input.clear();
        batch_frame.clear();
        
        for(int bi=0; bi< n_batch; ++bi){
            cap >> frame; 
            if(!frame.data) 
                break;
            height = frame.rows;
            width = frame.cols;
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        } 
        if(!frame.data) 
            break;
    
        //inference
        segNN.update(batch_dnn_input, n_batch);
        frame = segNN.draw();

        if(n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;
    }

    std::cout<<"segmentation end\n";   
    double mean = 0, mean_pre = 0, mean_post = 0;
    
    std::cout<<COL_GREENB<<"\n\nTime stats for size ["<<width<<","<<height<<"] :\n";
    // std::cout<<"Min: "<<*std::min_element(segNN.stats.begin(), segNN.stats.end())/n_batch<<" ms\n";    
    // std::cout<<"Max: "<<*std::max_element(segNN.stats.begin(), segNN.stats.end())/n_batch<<" ms\n";    
    
    for(int i=0; i<segNN.stats.size(); i++) mean += segNN.stats[i]; mean /= segNN.stats.size();
    for(int i=0; i<segNN.stats_pre.size(); i++) mean_pre += segNN.stats_pre[i]; mean_pre /= segNN.stats_pre.size();
    for(int i=0; i<segNN.stats_post.size(); i++) mean_post += segNN.stats_post[i]; mean_post /= segNN.stats_post.size();
    std::cout<<"Avg pre:\t"<<mean_pre/n_batch<<" ms\t"<<1000/(mean_pre/n_batch)<<" FPS\n";   
    std::cout<<"Avg inf:\t"<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n";   
    std::cout<<"Avg post:\t"<<mean_post/n_batch<<" ms\t"<<1000/(mean_post/n_batch)<<" FPS\n\n";   
    std::cout<<"Avg tot:\t"<<(mean_pre + mean_post + mean) /n_batch<<" ms\t"<<1000/((mean_pre + mean_post + mean)/n_batch)<<" FPS\n"<<COL_END;   
    

    return 0;
}

