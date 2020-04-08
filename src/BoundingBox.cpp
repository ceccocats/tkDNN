#include "BoundingBox.h"

namespace tk { namespace dnn {

float BoundingBox::overlap(const float p1, const float d1, const float p2, const float d2){
    float l1 = p1 - d1/2;
    float l2 = p2 - d2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = p1 + d1/2;
    float r2 = p2 + d2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float BoundingBox::boxesIntersection(const BoundingBox &b){
    float width = this->overlap(x, w, b.x, b.w);
    float height = this->overlap(y, h, b.y, b.h);
    if(width < 0 || height < 0) 
        return 0;
    float area = width*height;
    return area;
}

float BoundingBox::boxesUnion(const BoundingBox &b){
    float i = this->boxesIntersection(b);
    float u = w*h + b.w*b.h - i;
    return u;
}

float BoundingBox::IoU(const BoundingBox &b){
    float I = this->boxesIntersection(b);
    float U = this->boxesUnion(b);
    if (I == 0 || U == 0) 
        return 0;
    return I / U;
}

void BoundingBox::clear(){
    uniqueTruthIndex = -1;
    truthFlag = 0;
    maxIoU = 0;
}

std::ostream& operator<<(std::ostream& os, const BoundingBox& bb){
    os <<"w: "<< bb.w << ", h: "<< bb.h << ", x: "<< bb.x << ", y: "<< bb.y <<
         ", cat: "<< bb.cl << ", conf: "<< bb.prob<< ", truth: "<< 
         bb.truthFlag<< ", assignedGT: "<< bb.uniqueTruthIndex<< 
         ", maxIoU: "<< bb.maxIoU<<"\n";
    return os;
}

bool boxComparison (const BoundingBox& a,const BoundingBox& b) { 
    return (a.prob>b.prob); 
}

}}