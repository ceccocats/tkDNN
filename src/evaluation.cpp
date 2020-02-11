#include "evaluation.h"


void BoundingBox::clear()
{
    unique_truth_index = -1;
    truth_flag = 0;
    max_IoU = 0;
}

bool boxComparison (const BoundingBox& a,const BoundingBox& b) 
{ 
    return (a.prob>b.prob); 
}

std::ostream& operator<<(std::ostream& os, const BoundingBox& bb)
{
    os <<"w: "<< bb.w << ", h: "<< bb.h << ", x: "<< bb.x << ", y: "<< bb.y <<
         ", cat: "<< bb.cl << ", conf: "<< bb.prob<< ", truth: "<< 
         bb.truth_flag<< ", assignedGT: "<< bb.unique_truth_index<< 
         ", maxIoU: "<< bb.max_IoU<<"\n";
    return os;
}

void Frame::print() const
{
    std::cout<<"labels filename: "<<l_filename<<std::endl;
    std::cout<<"image filename: "<<i_filename<<std::endl;
    std::cout<<"GT: "<<std::endl;
    for(auto g: gt) std::cout<<g;
    std::cout<<"DET: "<<std::endl;
    for(auto d: det) std::cout<<d;
}

void PR::print()
{
    std::cout<<"precision: "<<precision<<" recall: "<<recall<<" tp: "<<tp<<" fp:"<<fp<<" fn:"<<fn<<std::endl;
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

float boxIntersection(const BoundingBox &a, const BoundingBox &b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) 
        return 0;
    float area = w*h;
    return area;
}

float boxUnion(const BoundingBox &a, const BoundingBox &b)
{
    float i = boxIntersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float boxIoU(const BoundingBox &a, const BoundingBox &b)
{
    float I = boxIntersection(a, b);
    // std::cout<<"I: "<<I<<std::endl;
    float U = boxUnion(a, b);
    // std::cout<<"U: "<<U<<std::endl;
    if (I == 0 || U == 0) 
        return 0;
    return I / U;
}

void readParams(char* config_filename, int& classes, int& map_points, 
                int& map_levels, float& map_step, float& IoU_thresh, 
                float& conf_thresh, bool& verbose)
{
    YAML::Node config   = YAML::LoadFile(config_filename);
    classes     = config["classes"].as<int>();
    map_points  = config["map_points"].as<int>();
    map_levels  = config["map_levels"].as<int>();
    map_step    = config["map_step"].as<float>();
    IoU_thresh  = config["IoU_thresh"].as<float>();
    conf_thresh = config["conf_thresh"].as<float>();
    verbose     = config["verbose"].as<bool>();

}

/* Credits to https://github.com/AlexeyAB/darknet/blob/master/src/detector.c*/
double computeMap(std::vector<Frame> &images,const int classes,const float IoU_thresh, const float conf_thresh, const int map_points, const bool verbose)
{
    if(verbose)
        for(auto img:images)
            img.print();

    int detections_count = 0;
    int groundtruths_count = 0;
    int unique_truth_count = 0;
    std::vector<int> truth_classes_count(classes,0);
    std::vector<int> dets_classes_count(classes,0);

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

    if(verbose)
    {
        std::cout<<"gt_count: "<<groundtruths_count<<std::endl;
        std::cout<<"det_count: "<<detections_count<<std::endl;
    }

    std::vector<BoundingBox> all_dets;
    std::vector<BoundingBox> all_gts;
    
    int gt_checked = 0;

    // for each detection comput IoU with groundtruth and match detetcion and 
    // groundtruth with IoU greater than IoU_thresh
    for(auto &img:images)
    {
        for(size_t i=0; i<img.det.size(); i++)
        {
            if(img.det[i].prob > conf_thresh)
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

        if(verbose)
            std::cout<<"Class: "<<i<<" AP: "<< avg_precision<<std::endl;
        mean_average_precision += avg_precision;
    }

    mean_average_precision = mean_average_precision / classes;

    std::cout<<"Classes: "<<classes<<" mAP " <<IoU_thresh<<":\t"<< mean_average_precision<<std::endl;
    return mean_average_precision;
}

double computeMapNIoULevels(std::vector<Frame> &images,const int classes,const float i_IoU_thresh, const float conf_thresh, const int map_points, const float map_step, const int map_levels, const bool verbose)
{
    double AP = 0;
    float IoU_thresh = i_IoU_thresh;
    for(int i=0; i<map_levels; ++i)
    {
        for(auto& img:images)
            for(auto & d:img.det)
                d.clear();
        AP += computeMap(images,classes,IoU_thresh,conf_thresh,map_points, verbose);
        IoU_thresh +=map_step;
    }
    AP/=map_levels;
    return AP;
}

void computeTPFPFN(std::vector<Frame> &images,const int classes,const float IoU_thresh, const float conf_thresh, bool verbose)
{
    std::vector<int> truth_classes_count(classes,0);
    std::vector<int> dets_classes_count(classes,0);
    std::vector<PR> pr( classes);
    
    for(auto &img:images)
    {
        for(auto& tc: truth_classes_count)
            tc = 0;
        for(auto& dc: dets_classes_count)
            dc = 0;

        std::vector<bool> det_assigned(img.det.size(), false);
        for(size_t j=0; j<img.gt.size(); j++)
        {
            truth_classes_count[img.gt[j].cl]++;
            float maxIoU = 0;
            int det_index = -1;
            for(size_t i=0; i<img.det.size(); i++)
            {
                if(img.det[i].prob > conf_thresh)
                {               
                    float currentIoU = boxIoU(img.det[i], img.gt[j]);
                    if(currentIoU > maxIoU && img.det[i].cl == img.gt[j].cl && !det_assigned[i])
                    {
                        maxIoU = currentIoU;                    
                        det_index = i;
                    }
                }
            }
            if(det_index > -1 && maxIoU > IoU_thresh && !det_assigned[det_index])
            {
                img.det[det_index].unique_truth_index = j;
                img.det[det_index].truth_flag = 1;
                img.det[det_index].max_IoU = maxIoU;
                det_assigned[det_index] = true;
                dets_classes_count[img.det[det_index].cl]++;
            }
        }

        for(size_t i=0; i<img.det.size(); i++)
        {
            if(img.det[i].truth_flag)
                pr[img.det[i].cl].tp ++;
            else
                pr[img.det[i].cl].fp ++;
        }
        for(size_t i=0; i<classes; i++)
        {
            pr[i].fn += truth_classes_count[i] - dets_classes_count[i];
        }
    }

    double avg_precision = 0, avg_recall = 0, f1_score = 0;

    for(size_t i=0; i<classes; i++)
    {
        pr[i].precision = (pr[i].tp + pr[i].fp) > 0 ? (double)pr[i].tp / (double)(pr[i].tp +pr[i].fp) : 0;
        pr[i].recall = (pr[i].tp + pr[i].fn) > 0 ? (double)pr[i].tp / (double)(pr[i].tp +pr[i].fn) : 0;
        if(verbose)
            std::cout<<"Class "<<i<<"\tTP: "<<pr[i].tp<<"\tFP: "<<pr[i].fp<<"\tFN: "<<pr[i].fn<<"\tprecision: "<<pr[i].precision<<"\trecall: "<<pr[i].recall<<std::endl;
            // std::cout<<i<<"\t"<<pr[i].tp<<"\t"<<pr[i].fp<<"\t"<<pr[i].fn<<"\t"<<pr[i].precision<<"\t"<<pr[i].recall<<std::endl;
        avg_precision += pr[i].precision;
        avg_recall += pr[i].recall;
    }
    avg_precision /= classes;
    avg_recall /= classes;

    f1_score = avg_precision + avg_recall > 0 ? 2 * ( avg_precision * avg_recall ) / ( avg_precision + avg_recall ) : 0;

    std::cout<<"avg precision: "<<avg_precision<<"\tavg recall: "<<avg_recall<<"\tavg f1 score:"<<f1_score<<std::endl;

    
}