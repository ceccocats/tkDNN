#include "evaluation.h"
#include <fstream>

namespace tk { namespace dnn {

void Frame::print() const{
    std::cout<<"labels filename: "<<lFilename<<std::endl;
    std::cout<<"image filename: "<<iFilename<<std::endl;
    std::cout<<"GT: "<<std::endl;
    for(auto g: gt) std::cout<<g;
    std::cout<<"DET: "<<std::endl;
    for(auto d: det) std::cout<<d;
}

void PR::print(){
    std::cout<<"precision: "<<precision<<" recall: "<<recall<<" tp: "<<tp<<" fp:"<<fp<<" fn:"<<fn<<std::endl;
}

void readmAPParams( const char* config_filename, int& classes, int& map_points, 
                    int& map_levels, float& map_step, float& IoU_thresh, 
                    float& conf_thresh, bool& verbose) {
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
double computeMap(  std::vector<Frame> &images,const int classes, 
                    const float IoU_thresh, const float conf_thresh, 
                    const int map_points, const bool verbose) {
    if(verbose)
        for(auto img:images)
            img.print();

    int detections_count = 0;
    int groundtruths_count = 0;
    int unique_truth_count = 0;
    std::vector<int> truth_classes_count(classes,0);
    std::vector<int> dets_classes_count(classes,0);

    //count groundtruth and detections in total and for each class
    for(auto i:images){
        for(auto gt:i.gt)
            truth_classes_count[gt.cl]++;
        for(auto det:i.det)
            dets_classes_count[det.cl]++;
        detections_count += i.det.size();
        groundtruths_count += i.gt.size();
    }

    if(verbose){
        std::cout<<"gt_count: "<<groundtruths_count<<std::endl;
        std::cout<<"det_count: "<<detections_count<<std::endl;
    }

    std::vector<BoundingBox> all_dets;
    std::vector<BoundingBox> all_gts;
    
    int gt_checked = 0;

    // for each detection compute IoU with groundtruth and match detetcion and 
    // groundtruth with IoU greater than IoU_thresh
    for(auto &img:images){
        for(size_t i=0; i<img.det.size(); i++){
            if(img.det[i].prob > conf_thresh){
                float maxIoU = 0;
                int truth_index = -1;
                for(size_t j=0; j<img.gt.size(); j++){
                    float currentIoU = img.det[i].IoU(img.gt[j]);
                    if(currentIoU > maxIoU && img.det[i].cl == img.gt[j].cl){
                        maxIoU = currentIoU;                    
                        truth_index = j;
                    }
                }
                if(truth_index > -1 && maxIoU > IoU_thresh){
                    img.det[i].uniqueTruthIndex = truth_index + gt_checked;
                    img.det[i].truthFlag = 1;
                    img.det[i].maxIoU = maxIoU;
                }
            }

            all_dets.push_back(img.det[i]);
        }
        gt_checked += img.gt.size();
    }

    if(verbose){
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
    for(int rank = 0; rank< detections_count; ++rank){
        if (rank > 0) {
            for (int class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }        

        //if it was detected and never detected before
        if (all_dets[rank].truthFlag == 1 && truth_flags[all_dets[rank].uniqueTruthIndex] == 0) {
            truth_flags[all_dets[rank].uniqueTruthIndex] = 1;
            pr[all_dets[rank].cl][rank].tp++;    // true-positive
        }
        else {
            pr[all_dets[rank].cl][rank].fp++;    // false-positive
        }

        for (int i = 0; i < classes; ++i){
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

            if (rank == (detections_count - 1) && dets_classes_count[i] != (tp + fp)) {    
                // check for last rank
                printf(" class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n", i, dets_classes_count[i], tp+fp, tp, fp);
            }
        }
    }

    if(verbose){
        for(int i=0; i < pr.size(); i++)   {
            std::cout<<"---------Class "<<i<<std::endl;
            for(auto r:pr[i])
                r.print();
        }
    }

    //compute average precision for each class. Two methods are available, 
    //based on map_points required
    double mean_average_precision = 0;
    double last_recall, last_precision, delta_recall;
    double cur_recall, cur_precision;
    double avg_precision = 0;
    for (int i = 0; i < classes; ++i) {
        avg_precision = 0;
        
        if (map_points == 0){ //mAP calculation: ImageNet, PascalVOC 2010-2012
            last_recall = pr[i][detections_count - 1].recall;
            last_precision = pr[i][detections_count - 1].precision;
            for (int rank = detections_count - 2; rank >= 0; --rank){
                delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;

                if (pr[i][rank].precision > last_precision) 
                    last_precision = pr[i][rank].precision;

                avg_precision += delta_recall * last_precision;
            }
        }
        else {//MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
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

double computeMapNIoULevels(std::vector<Frame> &images,const int classes, 
                            const float i_IoU_thresh, const float conf_thresh, 
                            const int map_points, const float map_step, 
                            const int map_levels, const bool verbose, 
                            const bool write_on_file, std::string net) {
    std::ofstream out_file;
    if(write_on_file){
	out_file.open("map.csv", std::ios_base::app);
        out_file<<net<<";";
    }

    double AP = 0, cur_AP = 0;
    float IoU_thresh = i_IoU_thresh;

    for(int i=0; i<map_levels; ++i){
        //clear detection-grounthuth matching
        for(auto& img:images)
            for(auto & d:img.det)
                d.clear();
        //compute mAP for the new IoU threshold
        cur_AP = computeMap(images,classes,IoU_thresh,conf_thresh,map_points, verbose);
	    if(write_on_file)
	        out_file<<cur_AP<<";";
	    AP += cur_AP;
            IoU_thresh +=map_step;
    }
    AP/=map_levels;

    if(write_on_file){
	    out_file<<AP<<"\n";
	    out_file.close();
    }
    return AP;
}

void computeTPFPFN( std::vector<Frame> &images,const int classes,    
                    const float IoU_thresh, const float conf_thresh, 
                    bool verbose, const bool write_on_file, std::string net) {

    std::ofstream out_file;
    if(write_on_file){
	    out_file.open("pr.csv", std::ios_base::app);
        out_file<<net<<";";
    }

    std::vector<int> truth_classes_count(classes,0);
    std::vector<int> dets_classes_count(classes,0);
    std::vector<PR> pr(classes);
    
    //compute TP, FP, FN for each image, for each class
    for(auto &img:images){
        for(auto& tc: truth_classes_count) tc = 0;
        for(auto& dc: dets_classes_count) dc = 0;

        std::vector<bool> det_assigned(img.det.size(), false);
        for(size_t j=0; j<img.gt.size(); j++){
            truth_classes_count[img.gt[j].cl]++;
            float maxIoU = 0;
            int det_index = -1;
            for(size_t i=0; i<img.det.size(); i++){
                if(img.det[i].prob > conf_thresh){               
                    float currentIoU = img.det[i].IoU(img.gt[j]);
                    if(currentIoU > maxIoU && img.det[i].cl == img.gt[j].cl && !det_assigned[i]){
                        maxIoU = currentIoU;                    
                        det_index = i;
                    }
                }
            }
            if(det_index > -1 && maxIoU > IoU_thresh && !det_assigned[det_index]){
                img.det[det_index].uniqueTruthIndex = j;
                img.det[det_index].truthFlag = 1;
                img.det[det_index].maxIoU = maxIoU;
                det_assigned[det_index] = true;
                dets_classes_count[img.det[det_index].cl]++;
            }
        }

        for(size_t i=0; i<img.det.size(); i++){
            if(img.det[i].truthFlag)
                pr[img.det[i].cl].tp ++;
            else
                pr[img.det[i].cl].fp ++;
        }
        for(size_t i=0; i<classes; i++){
            pr[i].fn += truth_classes_count[i] - dets_classes_count[i];
        }
    }

    //count all TP, FP, FN and compute precision, recall and f1-score
    double avg_precision = 0, avg_recall = 0, f1_score = 0;
    int TP = 0, FP = 0, FN = 0;
    for(size_t i=0; i<classes; i++){
        pr[i].precision = (pr[i].tp + pr[i].fp) > 0 ? (double)pr[i].tp / (double)(pr[i].tp +pr[i].fp) : 0;
        pr[i].recall = (pr[i].tp + pr[i].fn) > 0 ? (double)pr[i].tp / (double)(pr[i].tp +pr[i].fn) : 0;
        if(verbose)
            std::cout<<"Class "<<i<<"\tTP: "<<pr[i].tp<<"\tFP: "<<pr[i].fp<<"\tFN: "<<pr[i].fn<<"\tprecision: "<<pr[i].precision<<"\trecall: "<<pr[i].recall<<std::endl;
        avg_precision += pr[i].precision;
        avg_recall += pr[i].recall;

        TP += pr[i].tp;
        FP += pr[i].fp;
        FN += pr[i].fn;
    }
    avg_precision /= classes;
    avg_recall /= classes;

    f1_score = avg_precision + avg_recall > 0 ? 2 * ( avg_precision * avg_recall ) / ( avg_precision + avg_recall ) : 0;

    if(write_on_file){
        out_file<<TP<<";"<<FP<<";"<<FN<<";"<<avg_precision<<";"<<avg_recall<<";"<<f1_score<<"\n";
        out_file.close();
    }

    std::cout<<"avg precision: "<<avg_precision<<"\tavg recall: "<<avg_recall<<"\tavg f1 score:"<<f1_score<<std::endl;
}

void printJsonCOCOFormat(std::ofstream *out_file, const std::string image_path, std::vector<tk::dnn::box> bbox, const int classes, const int w, const int h)
{
    int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };
    std::string id = image_path.substr(image_path.find("images/")+7, image_path.find(".jpg") - image_path.find("images/") -7);
    int image_id = std::stoi(id);
    for (int i = 0; i < bbox.size(); ++i) {
        float xmin = bbox[i].x ;
        float xmax = bbox[i].x + float(bbox[i].w);
        float ymin = bbox[i].y;
        float ymax = bbox[i].y + float(bbox[i].h);

        //limit to image borders
        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        if(bbox[i].probs.size() == classes)
            for (int j = 0; j < classes; ++j) {
                //min threshold confidence is set in DetectionNN.h
                if (bbox[i].probs[j] > 0) {

                    *out_file <<    "{\"image_id\":" << image_id <<
                                    ", \"category_id\":" << coco_ids[j] << 
                                    ", \"bbox\":[" << bx << ", " << by << ", " << bw << ", " << bh << 
                                    "], \"score\":" << bbox[i].probs[j] << "},\n";
                }
            }
        else
            *out_file <<    "{\"image_id\":" << image_id <<
                            ", \"category_id\":" << coco_ids[bbox[i].cl] << 
                            ", \"bbox\":[" << bx << ", " << by << ", " << bw << ", " << bh << 
                            "], \"score\":" << bbox[i].prob << "},\n";
    }
}

}}
