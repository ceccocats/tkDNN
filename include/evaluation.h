#ifndef EVALUATION_H
#define EVALUATION_H_H

#include <iostream>
#include <vector>
#include <algorithm>

#include "tkdnn.h"


struct BoundingBox : public tk::dnn::box
{
    friend std::ostream& operator<<(std::ostream& os, const BoundingBox& bb);
    int unique_truth_index = -1;
    int truth_flag = 0;
    float max_IoU = 0;

    void clear();
};

std::ostream& operator<<(std::ostream& os, const BoundingBox& bb);
bool boxComparison (const BoundingBox& a,const BoundingBox& b) ;

struct Frame
{
    std::string l_filename;
    std::string i_filename;
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

float overlap(float x1, float w1, float x2, float w2);
float boxIntersection(const BoundingBox &a, const BoundingBox &b);
float boxUnion(const BoundingBox &a, const BoundingBox &b);
float boxIoU(const BoundingBox &a, const BoundingBox &b);

double computeMap(std::vector<Frame> &images,const int classes,const float IoU_thresh, const float conf_thresh=0.3, const int map_points=101, const bool verbose=false);
double computeMapNIoULevels(std::vector<Frame> &images,const int classes,const float i_IoU_thresh=0.5, const float conf_thresh=0.3, const int map_points=101, const float map_step=0.05, const int map_levels=10, const bool verbose=false);

void computeTPFPFN(std::vector<Frame> &images,const int classes,const float IoU_thresh=0.5, const float conf_thresh=0.3, bool verbose=false);

#endif /*EVALUATION_H*/