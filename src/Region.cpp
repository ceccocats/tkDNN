#include <iostream>

#ifdef OPENCV
    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/imgproc/imgproc.hpp>
#endif

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Region::Region(Network *net, int classes, int coords, int num) : 
    Layer(net) {    
    this->classes = classes;
    this->coords = coords;
    this->num = num;

    // same
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c;
    output_dim.h = input_dim.h;
    output_dim.w = input_dim.w;
    output_dim.l = input_dim.l;

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

Region::~Region() {
    checkCuda( cudaFree(dstData) );
}

int entry_index(int batch, int location, int entry, 
            int coords, int classes, dataDim_t &input_dim, dataDim_t &output_dim) {
    int n =   location / (input_dim.w*input_dim.h);
    int loc = location % (input_dim.w*input_dim.h);
    return batch*output_dim.tot() + n*input_dim.w*input_dim.h*(coords+classes+1) +
           entry*input_dim.w*input_dim.h + loc;
}

dnnType* Region::infer(dataDim_t &dim, dnnType* srcData) {

    checkCuda( cudaMemcpy(dstData, srcData, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));

    for (int b = 0; b < dim.n; ++b){
        for(int n = 0; n < num; ++n){
            int index = entry_index(b, n*dim.w*dim.h, 0, coords, classes, input_dim, output_dim);
            activationLOGISTICForward(srcData + index, dstData + index, 2*dim.w*dim.h);
            
            index = entry_index(b, n*dim.w*dim.h, coords, coords, classes, input_dim, output_dim);
            activationLOGISTICForward(srcData + index, dstData + index, dim.w*dim.h);
        }
    }

    //softmax start
    int index = entry_index(0, 0, coords + 1, coords, classes, input_dim, output_dim);
    softmaxForward(srcData + index, classes, output_dim.n*num, output_dim.tot()/num, 
                   output_dim.w*output_dim.h, 1, output_dim.w*output_dim.h, 1, dstData + index);

    dim = output_dim;
    return dstData;
}


/* Interpret class */
RegionInterpret::RegionInterpret(dataDim_t input_dim, dataDim_t output_dim, 
    int classes, int coords, int num, float thresh, std::string fname_weights) {

    this->input_dim = input_dim;
    this->output_dim = output_dim;

    this->classes = classes;
    this->coords = coords;
    this->num = num;
    this->thresh = thresh;
    this->res_boxes_n = 0;

    int tot = output_dim.w*output_dim.h*num;
    boxes = (box*)    malloc(tot*sizeof(box));
    probs = (float**) malloc(tot*sizeof(float *));
    for(int j = 0; j < tot; ++j) probs[j] = (float*) malloc((classes + 1)*sizeof(float *));
    s = (sortable_bbox*) malloc(tot*sizeof(sortable_bbox));

    //load anchors
    readBinaryFile(fname_weights, 2*num, &bias_h, &bias_d);
}

RegionInterpret::~RegionInterpret() {

    delete [] boxes;
    for(int j = 0; j < output_dim.w*output_dim.h*num; ++j) 
        delete [] probs[j];
    delete [] probs;
    delete [] s;

    delete [] bias_h;
    checkCuda( cudaFree(bias_d) );
}

box RegionInterpret::get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

void RegionInterpret::get_region_boxes(  float *input, int w, int h, int netw, int neth, float thresh, 
                                float **probs, box *boxes, int only_objectness, 
                                int *map, float tree_thresh, int relative) {
    int lh = output_dim.h;
    int lw = output_dim.w;
    float *predictions = input;
    for (int i = 0; i < lw*lh; ++i){

        int row = i / lw;
        int col = i % lw;
        for(int n = 0; n < num; ++n){

            int index = n*lw*lh + i;
            for(int j = 0; j < classes; ++j){
                probs[index][j] = 0;
            }
            int obj_index  = entry_index(0, n*lw*lh + i, 
                                         coords, coords, classes, output_dim, output_dim);
            int box_index  = entry_index(0, n*lw*lh + i, 0, 
                                         coords, classes, output_dim, output_dim);
            float scale = predictions[obj_index];
            boxes[index] = get_region_box(predictions, bias_h, n, box_index, col, row, lw, lh, lw*lh);

            float max = 0;
            for(int j = 0; j < classes; ++j){
                int class_index = entry_index(0, n*lw*lh + i, coords + 1 + j,
                                              coords, classes, output_dim, output_dim);
                float prob = scale*predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if(prob > max) max = prob;
            }
            probs[index][classes] = max;
        }
    }
    correct_region_boxes(boxes, lw*lh*num, w, h, netw, neth, relative);
}


void RegionInterpret::correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative) {
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}




//############################ BOX PROBABILITY UTILS ############################
int nms_comparator(const void *pa, const void *pb) {
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.cl] - b.probs[b.index][b.cl];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
float overlap(float x1, float w1, float x2, float w2) {
    /*
    //SLOW METHOD
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
    */

    //SPALLA METHOD
    float l;
    w1 < w2? l=w1 : l=w2;
    float d = fabs(x1 - x2);
    float k = fabs(w1 - w2)/2;
    if      (d <= k)    return l;
    else if (d <= k +l) return l - (d-k);
    else                return 0;
}
float box_intersection(box a, box b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    if(w <= 0) return 0;
    float h = overlap(a.y, a.h, b.y, b.h);
    if(h <= 0) return 0;
    float area = w*h;
    return area;
}
float box_union(box a, box b) {
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}
int max_index(float *a, int n) {
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}
//###############################################################################
float RegionInterpret::box_iou(box a, box b) {
    if(fabs(a.x - b.x) > (a.w+b.w)/2 || fabs(a.y - b.y) > (a.h+b.h)/2)
        return 0;
    return box_intersection(a, b)/box_union(a, b);
}



void RegionInterpret::interpretData(dnnType *data_h, int imageW, int imageH) {

    int imW, imH;
    if(imageW <= 0 || imageH <= 0) {
        imW = input_dim.w;
        imH = input_dim.h;
    } else {
        imW = imageW;
        imH = imageH;
    }

    int tot = output_dim.w*output_dim.h*num;

    get_region_boxes(data_h, imW, imH, output_dim.w, output_dim.h, thresh, probs, boxes, 0, 0, 0.5, 1);

    //delete repeats
    for(int i = 0; i < tot; ++i){
        s[i].index = i;       
        s[i].cl = classes;
        s[i].probs = probs;
    }
    qsort(s, tot, sizeof(sortable_bbox), nms_comparator);

    for(int i = 0; i < tot; ++i){
        if(probs[s[i].index][classes] == 0) continue;
        box a = boxes[s[i].index];
        for(int j = i+1; j < tot; ++j){
            box b = boxes[s[j].index];
            if (box_iou(a, b) > 0.3f){
                for(int k = 0; k < classes+1; ++k){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }

    res_boxes_n = 0;
    //print results
    for(int i = 0; i < tot; ++i){
        int cl = max_index(probs[i], classes);
        float prob = probs[i][cl];

        if(prob > thresh) {
            box b = boxes[i];
            int x = (b.x)*imW;
            int w = (b.w)*imW - b.x;
            int y = (b.y)*imH;
            int h = (b.h)*imH - b.y;

            //if(x < 0) x = 0;
            //if(y < 0) y = 0;
            //if(w > imW) w = imW;
            //if(h > imH) h = imH;

            //printf("%d: %.0f%% box(x1, y1, x2, y2): %d %d %d %d\n", cl, prob*100, x, y, w, h);
            b.x = x;
            b.y = y;
            b.h = h;
            b.w = w;
            b.cl = cl;
            b.prob = prob;
            res_boxes[res_boxes_n] = b;
            res_boxes_n++;
        }
    }
}

void RegionInterpret::showImageResult(dnnType *input_h) {

#ifdef OPENCV
    dataDim_t dim = input_dim;
    // read an image
    cv::Mat r(dim.h, dim.w, CV_32F, input_h);
    cv::Mat g(dim.h, dim.w, CV_32F, input_h + dim.h*dim.w);
    cv::Mat b(dim.h, dim.w, CV_32F, input_h + dim.h*dim.w*2);
    std::vector<cv::Mat> array_to_merge;
    array_to_merge.push_back(b);
    array_to_merge.push_back(g);
    array_to_merge.push_back(r);
    cv::Mat color;
    cv::merge(array_to_merge, color);

    for(int i=0; i<res_boxes_n; i++) {
        box bx = res_boxes[i];
        cv::rectangle(color, cv::Point(bx.x - bx.w/2, bx.y - bx.h/2), 
                             cv::Point(bx.x + bx.w/2, bx.y + bx.h/2),
        cv::Scalar( 0, 0, 255), 2);
    }
    cv::namedWindow("result");
    // show the image on window
    cv::imshow("result", color);
    // wait key for 5000 ms
    cv::waitKey(0);
#else
    std::cout<<"Visualization not supported, please recompile with OpenCV\n";
#endif
}

}}
