#include <iostream>

#ifdef OPENCV
    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/imgproc/imgproc.hpp>
#endif

#include "Layer.h"
#include "kernels.h"


namespace tk { namespace dnn {

Yolo::Yolo(Network *net, int classes, int num, std::string fname_weights, int n_masks, float scale_xy, double nms_thresh, nmsKind_t nsm_kind, int new_coords) : 
    Layer(net) {
    this->final = true;

    this->classes = classes;
    this->num = num;
    this->n_masks = n_masks;
    this->scaleXY = scale_xy;
    this->nms_thresh = nms_thresh;
    this->nsm_kind = nsm_kind;
    this->new_coords = new_coords;

    // load anchors
    if(fname_weights != "") {
        int seek = 0;
        readBinaryFile(fname_weights, n_masks, &mask_h, &mask_d, seek);
        seek += n_masks;
        readBinaryFile(fname_weights, n_masks*num*2, &bias_h, &bias_d, seek);
        //for(int i=0; i<n_masks*num*2; i++)
            //printf("%f\n", bias_h[i]);
    }

    // init default classes name
    classesNames.clear();
    for(int i=0; i<classes; i++) {
        classesNames.push_back(std::to_string(i));
    }

    // same
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c;
    output_dim.h = input_dim.h;
    output_dim.w = input_dim.w;
    output_dim.l = input_dim.l;

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
    predictions = nullptr;
}

Yolo::~Yolo() {
    checkCuda( cudaFree(dstData) );
}

int entry_index(int batch, int location, int entry, 
            int classes, dataDim_t &input_dim, dataDim_t &output_dim) {
    int n =   location / (input_dim.w*input_dim.h);
    int loc = location % (input_dim.w*input_dim.h);
    return batch*output_dim.tot() + n*input_dim.w*input_dim.h*(4+classes+1) +
           entry*input_dim.w*input_dim.h + loc;
}

Yolo::box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, int new_coords) {
    Yolo::box b;

    if(new_coords == 0){
        b.x = (i + x[index + 0*stride]) / lw;
        b.y = (j + x[index + 1*stride]) / lh;
        b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
        b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    }
    else{
        b.x = (i + x[index + 0 * stride] ) / lw;
        b.y = (j + x[index + 1 * stride] ) / lh;
        b.w = x[index + 2 * stride] * x[index + 2 * stride] * 4 * biases[2 * n] / w;
        b.h = x[index + 3 * stride] * x[index + 3 * stride] * 4 * biases[2 * n + 1] / h;
    }
    return b;
}

dnnType* Yolo::infer(dataDim_t &dim, dnnType* srcData) {

    checkCuda( cudaMemcpy(dstData, srcData, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));

    for (int b = 0; b < dim.n; ++b){
        for(int n = 0; n < n_masks; ++n){
            int index = entry_index(b, n*dim.w*dim.h, 0, classes, input_dim, output_dim);
            std::cout<<"new_coords"<<new_coords<<std::endl;
            if (new_coords == 1){
                if (this->scaleXY != 1) scalAdd(dstData + index, 2 * dim.w*dim.h, this->scaleXY, -0.5*(this->scaleXY - 1), 1);
            }
            else{
                activationLOGISTICForward(srcData + index, dstData + index, 2*dim.w*dim.h);

                if (this->scaleXY != 1) scalAdd(dstData + index, 2 * dim.w*dim.h, this->scaleXY, -0.5*(this->scaleXY - 1), 1);
                index = entry_index(b, n*dim.w*dim.h, 4, classes, input_dim, output_dim);
                activationLOGISTICForward(srcData + index, dstData + index, (1+classes)*dim.w*dim.h);
            }
        }
    }

    dim = output_dim;
    return dstData;
}

void correct_yolo_boxes(Yolo::detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
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
        Yolo::box b = dets[i].bbox;
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
        dets[i].bbox = b;
    }
}

int Yolo::computeDetections(Yolo::detection *dets, int &ndets, int netw, int neth, float thresh, int new_coords) {

    if(predictions == nullptr)
        predictions = new dnnType[output_dim.tot()];
    checkCuda( cudaMemcpy(predictions, dstData, output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost));

    int lw = output_dim.w;
    int lh = output_dim.h;

    if (output_dim.n == 2) {
        FatalError("BATCH of 2 not supported"); 
        //avg_flipped_yolo(l);
    }
    int i,j,n;
    int count = ndets;
    for (i = 0; i < lw*lh; ++i){
        int row = i / lw;
        int col = i % lw;
        for(n = 0; n < n_masks; ++n){
            int obj_index  = entry_index(0, n*lw*lh + i, 4, classes, input_dim, output_dim);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(0, n*lw*lh + i, 0, classes, input_dim, output_dim);
            
            dets[count].bbox = get_yolo_box(predictions, bias_h, mask_h[n], box_index, col, row, lw, lh, netw, neth, lw*lh, new_coords);
            dets[count].objectness = objectness;
            dets[count].classes = classes;
            for(j = 0; j < classes; ++j){
                int class_index = entry_index(0, n*lw*lh + i, 4 + 1 + j, classes, input_dim, output_dim);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            
            ++count;
            if(count >= MAX_DETECTIONS)
                FatalError("reach max boxes");
        }
    }

    correct_yolo_boxes(dets + ndets, count, netw, neth, netw, neth, 0);
    ndets = count;
    return count;
}

//////////////////////////////////////////////////////////////////
float yolo_overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float yolo_box_intersection(Yolo::box a, Yolo::box b)
{
    float w = yolo_overlap(a.x, a.w, b.x, b.w);
    float h = yolo_overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float yolo_box_union(Yolo::box a, Yolo::box b)
{
    float i = yolo_box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float yolo_box_iou(Yolo::box a, Yolo::box b)
{
    return yolo_box_intersection(a, b)/yolo_box_union(a, b);
}

void box_c(const Yolo::box a, const Yolo::box b, float& top, float& bot, float& left, float& right) {
    top = (std::min)(a.y - a.h / 2, b.y - b.h / 2);
    bot = (std::max)(a.y + a.h / 2, b.y + b.h / 2);
    left = (std::min)(a.x - a.w / 2, b.x - b.w / 2);
    right = (std::max)(a.x + a.w / 2, b.x + b.w / 2);
}

// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
float yolo_box_diou(const Yolo::box a, const Yolo::box b, const float nms_thresh=0.6)
{
    float top, bot, left, right;
    box_c(a, b, top, bot, left, right);
    float w = right - left;
    float h = bot - top;
    float c = w * w + h * h;
    float iou = yolo_box_iou(a, b);
    if (c == 0) 
        return iou;
    
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, nms_thresh);
    float diou_term = u;
    return iou - diou_term;
}

int yolo_nms_comparator(const void *pa, const void *pb)
{
    Yolo::detection a = *(Yolo::detection *)pa;
    Yolo::detection b = *(Yolo::detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
//////////////////////////////////////////////////////////////////7

Yolo::detection *Yolo::allocateDetections(int nboxes, int classes) {
    
    int i;
    Yolo::detection *dets = (Yolo::detection*) calloc(nboxes, sizeof(Yolo::detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = (float*) calloc(classes, sizeof(float));
    }
    return dets;
}

void Yolo::mergeDetections(Yolo::detection *dets, int ndets, int classes, double nms_thresh, nmsKind_t nsm_kind) {
    int total = ndets;

    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), yolo_nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (nsm_kind == GREEDY_NMS && yolo_box_iou(a, b) > nms_thresh)
                    dets[j].prob[k] = 0;
                else if (nsm_kind == DIOU_NMS && yolo_box_diou(a, b, nms_thresh) > nms_thresh)
                    dets[j].prob[k] = 0;
            }
        }
    }
}

}}
