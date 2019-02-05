#include <iostream>

#ifdef OPENCV
    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/imgproc/imgproc.hpp>
#endif

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

Yolo::detection *make_network_boxes(int nboxes, int classes) {
    
    int i;
    Yolo::detection *dets = (Yolo::detection*) calloc(nboxes, sizeof(Yolo::detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = (float*) calloc(classes, sizeof(float));
    }
    return dets;
}

Yolo::Yolo(Network *net, int classes, int num, const char* fname_weights) : 
    Layer(net) {
    
    this->classes = classes;
    this->num = num;

    // load anchors
    int seek = 0;
    readBinaryFile(fname_weights, num, &mask_h, &mask_d, seek);
    seek += num;
    readBinaryFile(fname_weights, 3*num*2, &bias_h, &bias_d, seek);

    printDeviceVector(num, mask_h, false);
    printDeviceVector(3*num*2, bias_h, false);

    // same
    output_dim.n = input_dim.n;
    output_dim.c = input_dim.c;
    output_dim.h = input_dim.h;
    output_dim.w = input_dim.w;
    output_dim.l = input_dim.l;

    std::cout<<"YOLO INPUT: ";
    input_dim.print();
    std::cout<<"\n";

    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
    predictions = nullptr;

    dets = make_network_boxes(MAX_DETECTIONS, classes);
    detected = 0;
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

Yolo::box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    Yolo::box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
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


dnnType* Yolo::infer(dataDim_t &dim, dnnType* srcData) {

    checkCuda( cudaMemcpy(dstData, srcData, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));

    for (int b = 0; b < dim.n; ++b){
        for(int n = 0; n < num; ++n){
            int index = entry_index(b, n*dim.w*dim.h, 0, classes, input_dim, output_dim);
            activationLOGISTICForward(srcData + index, dstData + index, 2*dim.w*dim.h);
            
            index = entry_index(b, n*dim.w*dim.h, 4, classes, input_dim, output_dim);
            activationLOGISTICForward(srcData + index, dstData + index, (1+classes)*dim.w*dim.h);
        }
    }

    dim = output_dim;
    return dstData;
}

int Yolo::computeDetections(int w, int h, int netw, int neth, float thresh) {

    if(predictions == nullptr)
        predictions = new dnnType[output_dim.tot()];
    checkCuda( cudaMemcpy(predictions, dstData, output_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost));

    int relative = 0;

    int lw = output_dim.w;
    int lh = output_dim.h;

    if (output_dim.n == 2) {
        FatalError("BATCH of 2 not supported"); 
        //avg_flipped_yolo(l);
    }
    int i,j,n;
    int count = 0;
    for (i = 0; i < lw*lh; ++i){
        int row = i / lw;
        int col = i % lw;
        for(n = 0; n < num; ++n){
            int obj_index  = entry_index(0, n*lw*lh + i, 4, classes, input_dim, output_dim);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(0, n*lw*lh + i, 0, classes, input_dim, output_dim);
            
            dets[count].bbox = get_yolo_box(predictions, bias_h, mask_h[n], box_index, col, row, lw, lh, netw, neth, lw*lh);
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
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);

    std::cout<<"DETECTED: "<<count<<"\n";
    detected = count;
    return count;
}

}}
