
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
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
    char *config_filename = "../demo/config.yaml";
    char * net = "yolo3.rt";
    char * labels_path = "../demo/COCO_val2017/all_labels.txt";
    bool show = false;
    bool write_dets = false;
    bool write_res_on_file = true;
    int n_images = 5000;

    bool verbose;
    int classes, map_points, map_levels;
    float map_step, IoU_thresh, conf_thresh;

    if(argc > 1)
        net = argv[1]; 
    if(argc > 2)
        ntype = argv[2][0];    
    if(argc > 3)
        labels_path = argv[3]; 
    if(argc > 4)
        config_filename = argv[4]; 

    if(!fileExist(config_filename))
        FatalError("Wrong config file path.");
    if(!fileExist(net))
        FatalError("Wrong net file path.");
    if(!fileExist(labels_path))
        FatalError("Wrong labels file path.");

    //read mAP parameters
    readParams( config_filename, classes,  map_points, map_levels, map_step,
                IoU_thresh, conf_thresh, verbose);

    std::ofstream times;
    if(write_res_on_file)
    {
        times.open ("times.csv", std::ios_base::app);
        times<<net<<";";
    }

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;

    tk::dnn::DetectionNN *detNN;  

    int n_classes = classes;   
    
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

    detNN->init(net, n_classes);

    std::ifstream all_labels(labels_path);
    std::string l_filename;
    std::vector<Frame> images;
    std::vector<tk::dnn::box> detected_bbox;

    std::cout<<"Reading groundtruth and generating detections"<<std::endl;

    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    for (int images_done=0 ; std::getline(all_labels, l_filename) && images_done < n_images ; ++images_done) 
    {
        std::cout <<COL_ORANGEB<< "Images done:\t" << images_done<< "\n"<<COL_END;

        Frame f;
        f.l_filename = l_filename;
        f.i_filename = l_filename;
        convertFilename(f.i_filename, "labels", "images", ".txt", ".jpg");

        // read frame
        if(!fileExist(f.i_filename.c_str()))
            FatalError("Wrong image file path.");

        cv::Mat frame = cv::imread(f.i_filename.c_str(), cv::IMREAD_COLOR);
        int height = frame.rows;
        int width = frame.cols;

        cv::Mat dnn_input;
        if(!frame.data) 
            break;
        dnn_input = frame.clone();

        //inference 
        
	    TIMER_START
        
        detected_bbox.clear();
        detNN->update(dnn_input);
        frame = detNN->draw(frame);
        detected_bbox = detNN->detected;
        
        TIMER_STOP

        if(write_res_on_file)
	        times<<t_ns<<";";

        std::ofstream myfile;
        if(write_dets)
            myfile.open ("det/"+f.l_filename.substr(l_filename.find("000")));

        // save detections labels
        for(auto d:detected_bbox)
        {
            //convert detected bb in the same format as label
            //<x_center>/<image_width> <y_center>/<image_width> <width>/<image_width> <height>/<image_width>
            BoundingBox b;
            b.x = (d.x + d.w/2) / width;
            b.y = (d.y + d.h/2) / height;
            b.w = d.w / width;
            b.h = d.h / height;
            b.prob = d.prob;
            b.cl = d.cl;
            f.det.push_back(b);

            if(write_dets)
                myfile << d.cl << " "<< d.prob << " "<< d.x << " "<< d.y << " "<< d.w << " "<< d.h <<"\n";

			if(show)// draw rectangle for detection
                cv::rectangle(frame, cv::Point(d.x, d.y), cv::Point(d.x + d.w, d.y + d.h), cv::Scalar(0, 0, 255), 2);             
        }

        if(write_dets)
            myfile.close();

        // read and save groundtruth labels
        std::ifstream labels(l_filename);
        for(std::string line; std::getline(labels, line); )
        {
            std::istringstream in(line); 
            BoundingBox b;
            in >> b.cl >> b.x >> b.y >> b.w >> b.h;  
            b.prob = 1;
            b.truth_flag = 1;
            f.gt.push_back(b);

            if(show)// draw rectangle for groundtruth
                cv::rectangle(frame, cv::Point((b.x-b.w/2)*width, (b.y-b.h/2)*height), cv::Point((b.x+b.w/2)*width,(b.y+b.h/2)*height), cv::Scalar(0, 255, 0), 2);             
        }    
      
        images.push_back(f);
        
        if(show)
        {
            cv::imshow("detection", frame);
            cv::waitKey(0);
        }
    }

    std::cout<<"Done."<<std::endl;

    
    
    //compute mAP
    double AP = computeMapNIoULevels(images,classes,IoU_thresh,conf_thresh, map_points, map_step, map_levels, verbose, write_res_on_file, net);
    std::cout<<"mAP "<<IoU_thresh<<":"<<IoU_thresh+map_step*(map_levels-1)<<" = "<<AP<<std::endl;

    //compute average precision, recall and f1score
    computeTPFPFN(images,classes,IoU_thresh,conf_thresh, verbose, write_res_on_file, net);

    if(write_res_on_file)
    {
	times<<"\n";
        times.close();
    }


    return 0;
}

