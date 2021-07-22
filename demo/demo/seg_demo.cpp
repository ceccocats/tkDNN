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

void writePred(const std::string& images_names, const std::string& gt_folder, const std::string& out_folder, tk::dnn::SegmentationNN& segNN, int& width, int& height, bool show=false){
    std::ifstream all_gt(images_names);
    std::string filename;
    cv::Mat frame;
    for (; std::getline(all_gt, filename); ) {
        std::cout<<filename<<std::endl;
        frame = cv::imread(gt_folder + filename);
        height = frame.rows;
        width = frame.cols;
        segNN.updateOriginal(frame, false);
        if(show)
            segNN.draw();
        cv::imwrite(out_folder + filename, segNN.segmented[0]);
    }
}

int main(int argc, char *argv[]) {

    std::cout<<"detection\n";
    signal(SIGINT, sig_handler);


    std::string net = "shelfnet_fp32.rt";
    if(argc > 1)
        net = argv[1]; 
    std::string input = "../demo/yolo_test.mp4";
    if(argc > 2)
        input = argv[2]; 
    int n_batch = 1;
    if(argc > 3)
        n_batch = atoi(argv[3]); 
    int n_classes = 19;
    if(argc > 4)
        n_classes = atoi(argv[4]); 
    bool resize = false;
    if(argc > 5)
        resize = atoi(argv[5]); 
    int baseline_resize = 1024;
    if(argc > 6)
        baseline_resize = atoi(argv[6]); 
    bool show = true;
    if(argc > 7)
        show = atoi(argv[7]); 
    bool write_pred = false;
    if(argc > 8)
        write_pred = atoi(argv[8]); 

    if(resize && (baseline_resize < 0 || baseline_resize > 5000)) 
        FatalError("Problem with baseline resize")
    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    //net initialization 
    tk::dnn::SegmentationNN segNN;
    segNN.init(net, n_classes, n_batch);

    int height = 0, width = 0;
    int basewidth=baseline_resize,  hsize; 
    
    if(write_pred){
        std::string gt_folder = "../demo/CityScapes_val/images/";
        std::string images_names = "../demo/CityScapes_val/all_images.txt";
        std::string out_folder = "seg/";

        writePred(images_names, gt_folder, out_folder, segNN, width, height, show);
    }
    else{
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
            int w,h;
            if(resize){
                w = basewidth;
                h = int((float(cap.get(cv::CAP_PROP_FRAME_HEIGHT))*float(basewidth/float(cap.get(cv::CAP_PROP_FRAME_WIDTH)))));
            }
            else{
                w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
                h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            }
            resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
        }

        cv::Mat frame;
        while(gRun) {
            cap >> frame; 
            if(!frame.data) 
                break;

            if(resize){
                hsize  = int((float(frame.rows)*float(basewidth/float(frame.cols))));
                cv::resize(frame, frame, cv::Size(basewidth, hsize));
            }

            height = frame.rows;
            width = frame.cols;

            //inference
            segNN.updateOriginal(frame, true);
            if(show)
                segNN.draw();

            if(SAVE_RESULT)
                resultVideo << segNN.segmented[0];
        }
    }

    std::cout<<"segmentation end\n";   
    double mean = 0, mean_pre = 0, mean_post = 0;
    
    std::cout<<COL_GREENB<<"\n\nTime stats for size ["<<width<<","<<height<<"] :\n";
    
    for(int i=0; i<segNN.stats.size(); i++) mean += segNN.stats[i]; mean /= segNN.stats.size();
    for(int i=0; i<segNN.stats_pre.size(); i++) mean_pre += segNN.stats_pre[i]; mean_pre /= segNN.stats_pre.size();
    for(int i=0; i<segNN.stats_post.size(); i++) mean_post += segNN.stats_post[i]; mean_post /= segNN.stats_post.size();
    std::cout<<"Avg pre:\t"<<mean_pre<<" ms\t"<<1000/(mean_pre)<<" FPS\n";   
    std::cout<<"Avg inf:\t"<<mean<<" ms\t"<<1000/(mean)<<" FPS\n";   
    std::cout<<"Avg post:\t"<<mean_post<<" ms\t"<<1000/(mean_post)<<" FPS\n\n";   
    std::cout<<"Avg tot:\t"<<(mean_pre + mean_post + mean) <<" ms\t"<<1000/((mean_pre + mean_post + mean))<<" FPS\n"<<COL_END;   
    
    return 0;
}

