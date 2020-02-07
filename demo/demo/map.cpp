
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

#include <map>

struct BoundigBox : public tk::dnn::box
{
    friend std::ostream& operator<<(std::ostream& os, const BoundigBox& bb);
    int unique_truth_index = -1;
    int truth_flag = 0;
    float max_IoU = 0;

    void clear()
    {
        unique_truth_index = -1;
        truth_flag = 0;
        max_IoU = 0;
    }
};

bool boxComparison (const BoundigBox& a,const BoundigBox& b) 
{ 
    return (a.prob>b.prob); 
}


std::ostream& operator<<(std::ostream& os, const BoundigBox& bb)
{
    os <<"w: "<< bb.w << ", h: "<< bb.h << ", x: "<< bb.x << ", y: "<< bb.y <<
         ", cat: "<< bb.cl << ", conf: "<< bb.prob<< ", truth: "<< 
         bb.truth_flag<< ", assignedGT: "<< bb.unique_truth_index<< 
         ", maxIoU: "<< bb.max_IoU<<"\n";
    return os;
}

struct Frame
{
    void print() const
    {
        std::cout<<"labels filename: "<<l_filename<<std::endl;
        std::cout<<"image filename: "<<i_filename<<std::endl;
        std::cout<<"GT: "<<std::endl;
        for(auto g: gt) std::cout<<g;
        std::cout<<"DET: "<<std::endl;
        for(auto d: det) std::cout<<d;
    }
    std::string l_filename;
    std::string i_filename;
    std::vector<BoundigBox> gt;
    std::vector<BoundigBox> det;
};

void convertFilename(std::string &filename,const std::string l_folder, const std::string i_folder, const std::string l_ext,const std::string i_ext)
{
    filename.replace(filename.find(l_folder),l_folder.length(),i_folder);
    filename.replace(filename.find(l_ext),l_ext.length(),i_ext);
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float boxIntersection(const BoundigBox &a, const BoundigBox &b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) 
        return 0;
    float area = w*h;
    return area;
}

float boxUnion(const BoundigBox &a, const BoundigBox &b)
{
    float i = boxIntersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float boxIoU(const BoundigBox &a, const BoundigBox &b)
{
    float I = boxIntersection(a, b);
    // std::cout<<"I: "<<I<<std::endl;
    float U = boxUnion(a, b);
    // std::cout<<"U: "<<U<<std::endl;
    if (I == 0 || U == 0) 
        return 0;
    return I / U;
}

struct PR 
{
    double precision = 0;
    double recall = 0;
    int tp = 0, fp = 0, fn = 0;

    void print()
    {
        std::cout<<"precision: "<<precision<<" recall: "<<recall<<" tp: "<<tp<<" fp:"<<fp<<" fn:"<<fn<<std::endl;
    }
};

double computeMap(std::vector<Frame> &images,const int classes,const float IoU_thresh, const int map_points, const bool verbose=false)
{
    std::cout<<"Computing mAP"<<std::endl;

    if(verbose)
        for(auto img:images)
            img.print();

    int detections_count = 0;
    int groundtruths_count = 0;
    int unique_truth_count = 0;
    std::vector<int> truth_classes_count(classes,0);
    std::vector<int> dets_classes_count(classes,0);
    // std::vector<int> avg_iou_per_class(classes,0);
    // std::vector<int> tp_for_thresh_per_class(classes,0);
    // std::vector<int> fp_for_thresh_per_class(classes,0);

    

    //count groundtruth and detections in total and for each class
    for(auto i:images)
    {
        for(auto gt:i.gt)
            truth_classes_count[gt.cl]++;
        for(auto det:i.det)
            dets_classes_count[det.cl]++;
        detections_count += i.det.size();
        groundtruths_count += i.gt.size();
    }

    std::cout<<"gt_count: "<<groundtruths_count<<std::endl;
    std::cout<<"det_count: "<<detections_count<<std::endl;

    std::vector<BoundigBox> all_dets;
    std::vector<BoundigBox> all_gts;
    
    int gt_checked = 0;

    // for each detection comput IoU with groundtruth and match detetcion and 
    // groundtruth with IoU greater than IoU_thresh
    for(auto &img:images)
    {
        for(size_t i=0; i<img.det.size(); i++)
        {
            if(img.det[i].prob > 0)
            {
                float maxIoU = 0;
                int truth_index = -1;
                for(size_t j=0; j<img.gt.size(); j++)
                {
                    float currentIoU = boxIoU(img.det[i], img.gt[j]);
                    if(currentIoU > maxIoU && img.det[i].cl == img.gt[j].cl)
                    {
                        maxIoU = currentIoU;                    
                        truth_index = j;
                    }
                }
                // std::cout<<"det i:"<<i<<" maxIoU:"<<maxIoU<<" tIndex:"<<truth_index<<std::endl;
                if(truth_index > -1 && maxIoU > IoU_thresh)
                {
                    // std::cout<<"(INSIDE) IoU thresh:"<<IoU_thresh<<" maxIoU:"<<maxIoU<<" maxIoU > IoU_thresh:"<<(maxIoU > IoU_thresh)<<std::endl;
                    img.det[i].unique_truth_index = truth_index + gt_checked;
                    img.det[i].truth_flag = 1;
                    img.det[i].max_IoU = maxIoU;
                }
            }

            all_dets.push_back(img.det[i]);
        }
        gt_checked += img.gt.size();
    }

    if(verbose)
    {
        for(auto img:images)
            img.print();
        std::cout<<"\n\n\n\n";
    }

    //sort all detections by descending value of confidence
    std::sort(all_dets.begin(), all_dets.end(), boxComparison);
    std::vector<int> truth_flags(groundtruths_count,0);

    if(verbose)
        for(auto d:all_dets)
            std::cout<<d;

    //compute precision-recall curve
    std::vector<std::vector<PR>> pr( classes, std::vector<PR>(detections_count));
    for(int rank = 0; rank< detections_count; ++rank)
    {
        if (rank > 0) 
        {
            for (int class_id = 0; class_id < classes; ++class_id) 
            {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }        

        //if it was detected and never detected before
        if (all_dets[rank].truth_flag == 1 && truth_flags[all_dets[rank].unique_truth_index] == 0) 
        {
            truth_flags[all_dets[rank].unique_truth_index] = 1;
            pr[all_dets[rank].cl][rank].tp++;    // true-positive
        }
        else 
        {
            pr[all_dets[rank].cl][rank].fp++;    // false-positive
        }

        for (int i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) 
                pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else 
                pr[i][rank].precision = 0;

            if ((tp + fn) > 0) 
                pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else 
                pr[i][rank].recall = 0;

            if (rank == (detections_count - 1) && dets_classes_count[i] != (tp + fp)) 
            {    // check for last rank
                printf(" class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n", i, dets_classes_count[i], tp+fp, tp, fp);
            }
        }
    }

    if(verbose)
    {
        for(int i=0; i < pr.size(); i++)   
        {
            std::cout<<"---------Class "<<i<<std::endl;
            for(auto r:pr[i])
                r.print();
        }
    }

    //compute average precision for each class. Two methods are avaible, 
    //based on map_points required
    double mean_average_precision = 0;
    double last_recall, last_precision, delta_recall;
    double cur_recall, cur_precision;
    double avg_precision = 0;
    for (int i = 0; i < classes; ++i) 
    {
        avg_precision = 0;
        
        if (map_points == 0) //mAP calculation: ImageNet, PascalVOC 2010-2012
        {
            last_recall = pr[i][detections_count - 1].recall;
            last_precision = pr[i][detections_count - 1].precision;
            for (int rank = detections_count - 2; rank >= 0; --rank)
            {
                delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;

                if (pr[i][rank].precision > last_precision) 
                    last_precision = pr[i][rank].precision;

                avg_precision += delta_recall * last_precision;
            }
        }
        else //MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
        {
            for (int point = 0; point < map_points; ++point) {
                cur_recall = point * 1.0 / ( map_points - 1 );
                cur_precision = 0;
                for (int rank = 0; rank < detections_count; ++rank)
                    if (pr[i][rank].recall >= cur_recall && pr[i][rank].precision > cur_precision) 
                        cur_precision = pr[i][rank].precision;

                avg_precision += cur_precision;
            }
            avg_precision = avg_precision / map_points;
        }

        std::cout<<"Class: "<<i<<" AP: "<< avg_precision<<std::endl;
        mean_average_precision += avg_precision;
    }

    mean_average_precision = mean_average_precision / classes;

    std::cout<<"Classes: "<<classes<<" mAP " <<IoU_thresh<<": "<< mean_average_precision<<std::endl;
    return mean_average_precision;
}

enum networkType_t { YOLO, CENTERNET};

int main(int argc, char *argv[]) 
{
    // char *net = "resnet101_cnet_FP32.rt";
    char *net = "yolo3.rt";
    if(argc > 1)
        net = argv[1]; 
    char type = 'y';
    if(argc > 2)
        type = argv[2][0]; 
    char *labels_path = "/media/887E650E7E64F67A/val2017/all_labels2017.txt";
    if(argc > 3)
        labels_path = argv[3]; 
    

    networkType_t ntype;
    switch(type)
    {
        case 'y':
            ntype = YOLO;
            break;
        case 'c':
            ntype = CENTERNET;
            break;
        default:
        FatalError("type not allowed (3rd parameter)");
    }

    bool show = false;

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;

    switch(ntype)
    {
        case YOLO:
            yolo.init(net);
            break;
        case CENTERNET:
            cnet.init(net);
            break;
        default:
            FatalError("Network type not allowed ");
    }

    std::ifstream all_labels(labels_path);
    std::string l_filename;
    std::vector<Frame> images;

    std::cout<<"Reading groundtruth and generating detections"<<std::endl;

    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    std::vector<tk::dnn::box> detected_bbox;
    
    int i=0;
    while (std::getline(all_labels, l_filename)) // && i < 1000) 
    {
        Frame f;
        f.l_filename = l_filename;
        f.i_filename = l_filename;
        convertFilename(f.i_filename, "labels", "images", ".txt", ".jpg");
        std::cout << f.i_filename << std::endl;
        std::cout << "images done:" << i++ << "\n";

        // generate detections
        cv::Mat frame = cv::imread(f.i_filename.c_str(), cv::IMREAD_COLOR);
        int height = frame.rows;
        int width = frame.cols;

        cv::Mat dnn_input;
        if(!frame.data) 
            break;
        dnn_input = frame.clone();

        //inference

        detected_bbox.clear();
        
        switch(ntype)
        {
            case YOLO:
                yolo.update(dnn_input);
                detected_bbox = yolo.detected;
                break;
            case CENTERNET:
                cnet.update(dnn_input);
                detected_bbox = cnet.detected;
                break;
            default:
                FatalError("Network type not allowed ");
        }

        // std::ofstream myfile;
        // myfile.open ("det/"+f.l_filename.substr(l_filename.find("000")));

        // save detections labels
        for(auto d:detected_bbox)
        {
            //convert detected bb in the same format as label
            //<x_center>/<image_width> <y_center>/<image_width> <width>/<image_width> <height>/<image_width>
            BoundigBox b;
            b.x = (d.x + d.w/2) / width;
            b.y = (d.y + d.h/2) / height;
            b.w = d.w / width;
            b.h = d.h / height;
            b.prob = d.prob;
            b.cl = d.cl;
            f.det.push_back(b);

            // myfile << d.cl << " "<< d.prob << " "<< d.x << " "<< d.y << " "<< d.w << " "<< d.h <<"\n";

			if(show)// draw rectangle for detection
                cv::rectangle(frame, cv::Point(d.x, d.y), cv::Point(d.x + d.w, d.y + d.h), cv::Scalar(0, 0, 255), 2);             
        }

        // myfile.close();

        // read and save groundtruth labels
        std::ifstream labels(l_filename);
        for(std::string line; std::getline(labels, line); )
        {
            std::istringstream in(line); 
            BoundigBox b;
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
    
    int classes = 80;
    int map_points = 0;
    int map_levels = 10;
    float map_step = 0.05;
    float IoU_thresh = 0.5;
    bool verbose = false;

    double AP = 0;
    for(int i=0; i<map_levels; ++i)
    {
        for(auto& img:images)
            for(auto & d:img.det)
                d.clear();
        AP += computeMap(images,classes,IoU_thresh,map_points, verbose);
        IoU_thresh +=map_step;
    }
    AP/=map_levels;
    std::cout<<"mAP "<<IoU_thresh-map_step*map_levels<<":"<<IoU_thresh<<" = "<<AP<<std::endl;


    return 0;
}

