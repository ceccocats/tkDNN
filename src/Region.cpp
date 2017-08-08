#include <iostream>

#include "Layer.h"
#include "kernels.h"

namespace tkDNN {

Region::Region(Network *net, int classes, int coords, int num, float thresh, const char* fname_weights) : 
    Layer(net) {

    this->classes = classes;
    this->coords = coords;
    this->num = num;
    this->thresh = thresh;
    
    // same
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c;
    output_dim.h = input_dim.h;
    output_dim.w = input_dim.w;
    output_dim.l = input_dim.l;

    //load anchors
    readBinaryFile(fname_weights, 2*num, &bias_h, &bias_d);

    checkCuda( cudaMalloc(&dstData, input_dim.tot()*sizeof(value_type)) );
}

Region::~Region() {
    checkCuda( cudaFree(dstData) );
}

int Region::entry_index(int batch, int location, int entry) {
    int n =   location / (input_dim.w*input_dim.h);
    int loc = location % (input_dim.w*input_dim.h);
    return batch*output_dim.tot() + n*input_dim.w*input_dim.h*(coords+classes+1) + entry*input_dim.w*input_dim.h + loc;
}

value_type* Region::infer(dataDim_t &dim, value_type* srcData) {

    checkCuda( cudaMemcpy(dstData, srcData, dim.tot()*sizeof(value_type), cudaMemcpyDeviceToDevice));

    for (int b = 0; b < dim.n; ++b){
        for(int n = 0; n < num; ++n){
            int index = entry_index(b, n*dim.w*dim.h, 0);
            activationLOGISTICForward(srcData + index, dstData + index, 2*dim.w*dim.h);
            
            index = entry_index(b, n*dim.w*dim.h, coords);
            activationLOGISTICForward(srcData + index, dstData + index, dim.w*dim.h);
        }
    }

    //softmax start
    int index = entry_index(0, 0, coords + 1);
    softmaxForward(srcData + index, classes, output_dim.n*num, output_dim.tot()/num, 
                   output_dim.w*output_dim.h, 1, output_dim.w*output_dim.h, 1, dstData + index);

    dim = output_dim;
    return dstData;
}


box Region::get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

void Region::get_region_boxes(  float *input, int w, int h, int netw, int neth, float thresh, 
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
            int obj_index  = entry_index(0, n*lw*lh + i, coords);
            int box_index  = entry_index(0, n*lw*lh + i, 0);
            float scale = predictions[obj_index];
            boxes[index] = get_region_box(predictions, bias_h, n, box_index, col, row, lw, lh, lw*lh);

            float max = 0;
            for(int j = 0; j < classes; ++j){
                int class_index = entry_index(0, n*lw*lh + i, coords + 1 + j);
                float prob = scale*predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if(prob > max) max = prob;
            }
            probs[index][classes] = max;
        }
    }
    correct_region_boxes(boxes, lw*lh*num, w, h, netw, neth, relative);
}


void Region::correct_region_boxes(box *boxes, int n, int w, int h, int netw, int neth, int relative) {
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
struct sortable_bbox {
    int index;
    int cl;
    float **probs;
};
int nms_comparator(const void *pa, const void *pb) {
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.cl] - b.probs[b.index][b.cl];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}
float box_intersection(box a, box b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}
float box_union(box a, box b) {
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}
float box_iou(box a, box b) {
    return box_intersection(a, b)/box_union(a, b);
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




void Region::interpretData() {

    int imW = 768, imH = 576;

    int tot = output_dim.w*output_dim.h*num;
    float *lel = new value_type[output_dim.tot()];
    cudaMemcpy(lel, dstData, output_dim.tot()*sizeof(value_type), cudaMemcpyDeviceToHost);
    box *boxes =  (box*) calloc(tot, sizeof(box));
    float **probs = (float**) calloc(tot, sizeof(float *));
    for(int j = 0; j < tot; ++j) probs[j] = (float*)calloc(classes + 1, sizeof(float *));

    get_region_boxes(lel, imW, imH, output_dim.w, output_dim.h, thresh, probs, boxes, 0, 0, 0.5, 1);

    //delete repeats
    sortable_bbox *s = (sortable_bbox*)calloc(tot, sizeof(sortable_bbox));
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
            if (box_iou(a, b) > thresh){
                for(int k = 0; k < classes+1; ++k){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
    //

    //print results
    for(int i = 0; i < tot; ++i){
        int cl = max_index(probs[i], classes);
        float prob = probs[i][cl];
        if(prob > thresh) {
            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            printf("%d: %.0f%%\n", cl, prob*100);
        }
    }
}

}
