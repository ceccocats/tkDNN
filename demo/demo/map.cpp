
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
    int n_batches = 1;
    float confidence_thresh = 0.3;
    bool show = false;
    bool write_dets = false;
    bool write_res_on_file = true;
    bool write_coco_json = false;
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
    if(argc > 5)
        n_batches = atoi(argv[5]); 
    if(argc > 6)
        confidence_thresh = atof(argv[6]); 

    std::cout<<"conf t: "<<confidence_thresh<<std::endl;

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
        times.open("times_"+net_name+"_"+ std::to_string(n_batches)+"_"+std::to_string(confidence_thresh)+".csv");
        memory.open("memory.csv", std::ios_base::app);
        memory<<net_name+"_"+ std::to_string(n_batches)+"_"+std::to_string(confidence_thresh)<<";";
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

    bool file_ok = false;

    int images_done;
    for (images_done=0 ; images_done < n_images ;) {
        

        int cur_batches = 0;
        std::vector<cv::Mat> batch_frames;
        std::vector<cv::Mat> batch_dnn_input;
            
        std::vector<tk::dnn::Frame> cur_frames;
        for(;cur_batches<n_batches && images_done < n_images;cur_batches++, ++images_done){

            std::getline(all_labels, l_filename);
            file_ok = all_labels ? true : false ;
            if (!file_ok)
                break;

            tk::dnn::Frame f;
            f.lFilename = l_filename;
            f.iFilename = l_filename;
            convertFilename(f.iFilename, "labels", "images", ".txt", ".jpg");

            // read frame
            if(!fileExist(f.iFilename.c_str()))
                FatalError("Wrong image file path.");

            cv::Mat frame = cv::imread(f.iFilename.c_str(), cv::IMREAD_COLOR);
            batch_frames.push_back(frame);
            f.height = frame.rows;
            f.width = frame.cols;

            if(!frame.data) 
                break;
            batch_dnn_input.push_back(frame.clone());

            // read and save groundtruth labels
            if(fileExist(f.lFilename.c_str()))
            {
                std::ifstream labels(f.lFilename);
                for(std::string line; std::getline(labels, line); ){
                    std::istringstream in(line); 
                    tk::dnn::BoundingBox b;
                    in >> b.cl >> b.x >> b.y >> b.w >> b.h;  
                    b.prob = 1;
                    b.truthFlag = 1;
                    f.gt.push_back(b);

                    if(show)// draw rectangle for groundtruth
                        cv::rectangle(batch_frames[cur_batches], cv::Point((b.x-b.w/2)*f.width, (b.y-b.h/2)*f.height), cv::Point((b.x+b.w/2)*f.width,(b.y+b.h/2)*f.height), cv::Scalar(0, 255, 0), 2);             
                }
            } 

            cur_frames.push_back(f);
        }
        if (!file_ok)
            break;

        //inference 
        detNN->update(batch_dnn_input,cur_batches,write_res_on_file, &times, write_coco_json);
        detNN->draw(batch_frames);

        for(int j=0;j<cur_frames.size(); ++j){
            if(write_coco_json)
            printJsonCOCOFormat(&coco_json, cur_frames[j].iFilename.c_str(), detNN->batchDetected[j], classes,  cur_frames[j].width, cur_frames[j].height);        

            std::ofstream myfile;
            if(write_dets)
                myfile.open ("det/"+cur_frames[j].lFilename.substr(cur_frames[j].lFilename.find("labels/") + 7));

            // save detections labels
            for(auto d:detNN->batchDetected[j]){
                //convert detected bb in the same format as label
                //<x_center>/<image_width> <y_center>/<image_width> <width>/<image_width> <height>/<image_width>
                tk::dnn::BoundingBox b;
                b.x = (d.x + d.w/2) / cur_frames[j].width;
                b.y = (d.y + d.h/2) / cur_frames[j].height;
                b.w = d.w / cur_frames[j].width;
                b.h = d.h / cur_frames[j].height;
                b.prob = d.prob;
                b.cl = d.cl;
                cur_frames[j].det.push_back(b);

                if(write_dets)
                    myfile << d.cl << " "<< d.prob << " "<< b.x << " "<< b.y << " "<< b.w << " "<< b.h <<"\n";

                if(show)// draw rectangle for detection
                    cv::rectangle(batch_frames[j], cv::Point(d.x, d.y), cv::Point(d.x + d.w, d.y + d.h), cv::Scalar(0, 0, 255), 2);             
                }

            if(write_dets)
                myfile.close();
      
            images.push_back(cur_frames[j]);
        
            if(show){
                cv::imshow("detection", batch_frames[j]);
                cv::waitKey(0);
            }
            
        }
        std::cout <<COL_ORANGEB<< "Images done:\t" << images_done<< "\tcur batch:\t"<<cur_batches<< "\n"<<COL_END;
        
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
    double AP = tk::dnn::computeMapNIoULevels(images,classes,IoU_thresh,confidence_thresh, map_points, map_step, map_levels, verbose, write_res_on_file, net_name+"_"+ std::to_string(n_batches)+"_"+std::to_string(confidence_thresh));
    std::cout<<"mAP "<<IoU_thresh<<":"<<IoU_thresh+map_step*(map_levels-1)<<" = "<<AP<<std::endl;

    //compute average precision, recall and f1score
    tk::dnn::computeTPFPFN(images,classes,IoU_thresh,confidence_thresh, verbose, write_res_on_file, net_name +"_"+ std::to_string(n_batches)+"_"+std::to_string(confidence_thresh));

    if(write_res_on_file){
        memory<<vm_total/images_done/1024.0<<";"<<rss_total/images_done/1024.0<<"\n";
        times.close();
        memory.close();
    }

    return 0;
}

