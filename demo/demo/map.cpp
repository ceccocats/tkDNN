
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#ifdef __linux__
#include <unistd.h>
#endif

#include <mutex>
#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Yolo3Detection.h"
#include "CenternetDetection.h"
#include "MobilenetDetection.h"

#include "evaluation.h"

#include <map>

void convertFilename(std::string &filename,const std::string l_folder, const std::string i_folder, const std::string l_ext,const std::string i_ext)
{
    filename.replace(filename.find(l_folder),l_folder.length(),i_folder);
    filename.replace(filename.find(l_ext),l_ext.length(),i_ext);
}

int main(int argc, char *argv[]) 
{
    char ntype = 'y';
    const char *config_filename = "../demo/config.yaml";
    const char * net = "yolo3.rt";
    const char * labels_path = "../demo/COCO_val2017/all_labels.txt";
    bool show = false;
    bool write_dets = false;
    bool write_res_on_file = true;
    bool write_coco_json = true;
    int n_images = 5000;

    bool verbose;
    int classes, map_points, map_levels;
    float map_step, IoU_thresh, conf_thresh;

    double vm_total = 0, rss_total = 0;
    double vm, rss;

    //read args
    if(argc > 1)
        net = argv[1]; 
    if(argc > 2)
        ntype = argv[2][0];    
    if(argc > 3)
        labels_path = argv[3]; 
    if(argc > 4)
        config_filename = argv[4]; 

    //check if files needed exist
    if(!fileExist(config_filename))
        FatalError("Wrong config file path.");
    if(!fileExist(net))
        FatalError("Wrong net file path.");
    if(!fileExist(labels_path))
        FatalError("Wrong labels file path.");

    //read mAP parameters
    tk::dnn::readmAPParams( config_filename, classes,  map_points, map_levels, map_step,
                IoU_thresh, conf_thresh, verbose);

    //extract network name from rt path
    std::string net_name;
    removePathAndExtension(net, net_name);
    std::cout<<"Network: "<<net_name<<std::endl;

    //open files (if needed)
    std::ofstream times, memory, coco_json;

    if(write_coco_json){
        coco_json.open(net_name+"_COCO_res.json");
        coco_json << "[\n";
    }

    if(write_res_on_file){
        times.open("times_"+net_name+".csv");
        memory.open("memory.csv", std::ios_base::app);
        memory<<net<<";";
    }

    // instantiate detector
    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;
    tk::dnn::DetectionNN *detNN;  
    int n_classes = classes;   
    switch(ntype){
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
    detNN->init(net, n_classes, 1, conf_thresh);

    //read images 
    std::ifstream all_labels(labels_path);
    std::string l_filename;
    std::vector<tk::dnn::Frame> images;
    std::vector<tk::dnn::box> detected_bbox;

    std::cout<<"Reading groundtruth and generating detections"<<std::endl;

    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    int images_done;
    for (images_done=0 ; std::getline(all_labels, l_filename) && images_done < n_images ; ++images_done) {
        std::cout <<COL_ORANGEB<< "Images done:\t" << images_done<< "\n"<<COL_END;

        tk::dnn::Frame f;
        f.lFilename = l_filename;
        f.iFilename = l_filename;
        convertFilename(f.iFilename, "labels", "images", ".txt", ".jpg");

        // read frame
        if(!fileExist(f.iFilename.c_str()))
            FatalError("Wrong image file path.");

        cv::Mat frame = cv::imread(f.iFilename.c_str(), cv::IMREAD_COLOR);
        std::vector<cv::Mat> batch_frames;
        batch_frames.push_back(frame);
        int height = frame.rows;
        int width = frame.cols;

        if(!frame.data) 
            break;
        std::vector<cv::Mat> batch_dnn_input;
        batch_dnn_input.push_back(frame.clone());

        //inference 
        detected_bbox.clear();
        detNN->update(batch_dnn_input,1,write_res_on_file, &times, write_coco_json);
        detNN->draw(batch_frames);
        detected_bbox = detNN->detected;

        if(write_coco_json)
            printJsonCOCOFormat(&coco_json, f.iFilename.c_str(), detected_bbox, classes,  width, height);        

        std::ofstream myfile;
        if(write_dets)
            myfile.open ("det/"+f.lFilename.substr(f.lFilename.find("labels/") + 7));

        // save detections labels
        for(auto d:detected_bbox){
            //convert detected bb in the same format as label
            //<x_center>/<image_width> <y_center>/<image_width> <width>/<image_width> <height>/<image_width>
            tk::dnn::BoundingBox b;
            b.x = (d.x + d.w/2) / width;
            b.y = (d.y + d.h/2) / height;
            b.w = d.w / width;
            b.h = d.h / height;
            b.prob = d.prob;
            b.cl = d.cl;
            f.det.push_back(b);

            if(write_dets)
                myfile << d.cl << " "<< d.prob << " "<< b.x << " "<< b.y << " "<< b.w << " "<< b.h <<"\n";

			if(show)// draw rectangle for detection
                cv::rectangle(batch_frames[0], cv::Point(d.x, d.y), cv::Point(d.x + d.w, d.y + d.h), cv::Scalar(0, 0, 255), 2);             
        }

        if(write_dets)
            myfile.close();

        // read and save groundtruth labels
        if(fileExist(f.lFilename.c_str()))
        {
            std::ifstream labels(l_filename);
            for(std::string line; std::getline(labels, line); ){
                std::istringstream in(line); 
                tk::dnn::BoundingBox b;
                in >> b.cl >> b.x >> b.y >> b.w >> b.h;  
                b.prob = 1;
                b.truthFlag = 1;
                f.gt.push_back(b);

                if(show)// draw rectangle for groundtruth
                    cv::rectangle(batch_frames[0], cv::Point((b.x-b.w/2)*width, (b.y-b.h/2)*height), cv::Point((b.x+b.w/2)*width,(b.y+b.h/2)*height), cv::Scalar(0, 255, 0), 2);             
            }
        }    
      
        images.push_back(f);
        
        if(show){
            cv::imshow("detection", batch_frames[0]);
            cv::waitKey(0);
        }

        getMemUsage(vm, rss);
        vm_total += vm;
        rss_total += rss;

        
    }

    if(write_coco_json){
        coco_json.seekp (coco_json.tellp() - std::streampos(2));
        coco_json << "\n]\n";
        coco_json.close();
    }

    std::cout << "Avg VM[MB]: " << vm_total/images_done/1024.0 << ";Avg RSS[MB]: " << rss_total/images_done/1024.0 << std::endl;

    //compute mAP
    double AP = tk::dnn::computeMapNIoULevels(images,classes,IoU_thresh,conf_thresh, map_points, map_step, map_levels, verbose, write_res_on_file, net_name);
    std::cout<<"mAP "<<IoU_thresh<<":"<<IoU_thresh+map_step*(map_levels-1)<<" = "<<AP<<std::endl;

    //compute average precision, recall and f1score
    tk::dnn::computeTPFPFN(images,classes,IoU_thresh,conf_thresh, verbose, write_res_on_file, net_name);

    if(write_res_on_file){
        memory<<vm_total/images_done/1024.0<<";"<<rss_total/images_done/1024.0<<"\n";
        times.close();
        memory.close();
    }

    return 0;
}

