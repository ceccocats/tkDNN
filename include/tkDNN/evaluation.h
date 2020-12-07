#ifndef EVALUATION_H
#define EVALUATION_H

#include <iostream>
#include <vector>
#include <algorithm>

#include <yaml-cpp/yaml.h>

#include "tkdnn.h"
#include "BoundingBox.h"

namespace tk { namespace dnn {

struct Frame
{
    std::string lFilename;
    std::string iFilename;
    std::vector<BoundingBox> gt;
    std::vector<BoundingBox> det;

    void print() const;
};

struct PR 
{
    double precision = 0;
    double recall = 0;
    int tp = 0, fp = 0, fn = 0;

    void print();
};

void readmAPParams( const char* config_filename, int& classes, int& map_points, 
                    int& map_levels, float& map_step, float& IoU_thresh, 
                    float& conf_thresh, bool& verbose);

/**
 * This method computes the mean Average Precision for a set of detections and 
 * groundtruths. It returns the mAP for a given IoU threshold, and a given 
 * confidence threshold over all the classes.
 *
 * @param images collection of frames on which to compute the metrics
 * @param classes number of classes of the considered dataset
 * @param IoU_thresh threshold used to compute Intersection over Union
 * @param conf_thresh threshold used to filter bounding boxes based on their 
 *                    confidence (or probability)
 * @param map_points number of point used to compute the mAP. if 0 is given, 
 *                  all the recall levels are evaluated, otherwise only 
 *                  map_point recall levels are used. For COCO evaluation 
 *                  101 points are used.
 * @param verbose is set to true, prints on screen additional info
 *
 * @return mAP computed
 */
double computeMap(  std::vector<Frame> &images,const int classes,
                    const float IoU_thresh, const float conf_thresh=0.3, 
                    const int map_points=101, const bool verbose=false);


/**
 * This method computes the mean Average Precision for a set of detections and 
 * groundtruths on several IoU thresholds. It is used to compute, for example, 
 * the most used metric in Object Detection, namely the mAP 0.5:0.95, which is 
 * the average among the mAP for IoU level from 0.5 to 0.95 with a step of 0.05.
 *
 * @param images collection of frames on which to compute the metrics
 * @param classes number of classes of the considered dataset
 * @param IoU_thresh starting threshold used to compute Intersection over Union
 * @param conf_thresh threshold used to filter bounding boxes based on their 
 *                    confidence (or probability)
 * @param map_points number of point used to compute the mAP. if 0 is given, 
 *                  all the recall levels are evaluated, otherwise only 
 *                  map_point recall levels are used. For COCO evaluation 
 *                  101 points are used.
 * @param map_step step used to increment IoU threshold
 * @param map_levels number of IoU step to perform
 * @param verbose is set to true, prints on screen additional info
 * @param write_on_file if set to true, the results produced by this function 
 *                      are written on file
 * @param net name of the considered neural network 
 *
 * @return  mAP IoU_tresh:IoU_tresh+map_step*map_levels (e.g. mAP 0.5:0.95 when 
 *          map_step=0.05 and map_levels=10)
 */
double computeMapNIoULevels(std::vector<Frame> &images,const int classes,
                const float i_IoU_thresh=0.5, const float conf_thresh=0.3, 
                const int map_points=101, const float map_step=0.05, 
                const int map_levels=10, const bool verbose=false, 
                const bool write_on_file = false, std::string net = "");
/**
 * This method computes the number of True Positive (TP), False Positive (FP),
 * False Negative (FN), precision, recall and f1-score.
 * Those values are computer over all the detections, over all the classes.
 *
 * @param images collection of frames on which to compute the metrics
 * @param classes number of classes of the considered dataset
 * @param IoU_thresh threshold used to compute Intersection over Union
 * @param conf_thresh threshold used to filter bounding boxes based on their 
 *                    confidence (or probability)
 * @param verbose is set to true, prints on screen additional info
 * @param write_on_file if set to true, the results produced by this function 
 *                      are written on file
 * @param net name of the considered neural network 
 */
void computeTPFPFN( std::vector<Frame> &images,const int classes,
                    const float IoU_thresh=0.5, const float conf_thresh=0.3, 
                    bool verbose=false, const bool write_on_file=false,    
                    std::string net="");


void printJsonCOCOFormat(std::ofstream *out_file, const std::string image_path, std::vector<tk::dnn::box> bbox, const int classes, const int w, const int h);

}}
#endif /*EVALUATION_H*/

