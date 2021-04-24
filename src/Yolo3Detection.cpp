#include "Yolo3Detection.h"


namespace tk { namespace dnn {

bool Yolo3Detection::init(const std::string& tensor_path, const int n_classes, const int n_batches, const float conf_thresh) {

    //convert network to tensorRT
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );

    nBatches = n_batches;
    confThreshold = conf_thresh;
    tk::dnn::dataDim_t idim = netRT->input_dim;    
    idim.n = nBatches;

    if(netRT->pluginFactory->n_yolos < 2 ) {
        FatalError("this is not yolo3");
    }

    for(int i=0; i<netRT->pluginFactory->n_yolos; i++) {
        YoloRT *yRT = netRT->pluginFactory->yolos[i];
        classes = yRT->classes;
        num = yRT->num;
        nMasks = yRT->n_masks;

        // make a yolo layer to interpret predictions
        yolo[i] = new tk::dnn::Yolo(nullptr, classes, nMasks, ""); // yolo without input and bias
        yolo[i]->mask_h = new dnnType[nMasks];
        yolo[i]->bias_h = new dnnType[num*nMasks*2];
        memcpy(yolo[i]->mask_h, yRT->mask, sizeof(dnnType)*nMasks);
        memcpy(yolo[i]->bias_h, yRT->bias, sizeof(dnnType)*num*nMasks*2);
        yolo[i]->input_dim = yolo[i]->output_dim = tk::dnn::dataDim_t(1, yRT->c, yRT->h, yRT->w);
        yolo[i]->classesNames = yRT->classesNames;
        yolo[i]->nms_thresh = yRT->nms_thresh;
        yolo[i]->nsm_kind = (tk::dnn::Yolo::nmsKind_t) yRT->nms_kind;
        yolo[i]->new_coords = yRT->new_coords;
    }

    dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);
#ifndef OPENCV_CUDACONTRIB
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*idim.tot()));
#endif
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*idim.tot()));

    // class colors precompute    
    for(int c=0; c<classes; c++) {
        int offset = c*123457 % classes;
        float r = getColor(2, offset, classes);
        float g = getColor(1, offset, classes);
        float b = getColor(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }

    classesNames = getYoloLayer()->classesNames;
    return true;
} 

void Yolo3Detection::preprocess(cv::Mat &frame, const int bi){
#ifdef OPENCV_CUDACONTRIB
    cv::cuda::GpuMat orig_img, img_resized;
    orig_img = cv::cuda::GpuMat(frame);
    cv::cuda::resize(orig_img, img_resized, cv::Size(netRT->input_dim.w, netRT->input_dim.h));

    img_resized.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    //split channels
    cv::cuda::split(imagePreproc,bgr);//split source

    //write channels
    for(int i=0; i<netRT->input_dim.c; i++) {
        int size = imagePreproc.rows * imagePreproc.cols;
        int ch = netRT->input_dim.c-1 -i;
        bgr[ch].download(bgr_h); //TODO: don't copy back on CPU
        checkCuda( cudaMemcpy(input_d + i*size + netRT->input_dim.tot()*bi, (float*)bgr_h.data, size*sizeof(dnnType), cudaMemcpyHostToDevice));
    }
#else
    cv::resize(frame, frame, cv::Size(netRT->input_dim.w, netRT->input_dim.h));
    frame.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    //split channels
    cv::split(imagePreproc,bgr);//split source

    //write channels
    for(int i=0; i<netRT->input_dim.c; i++) {
        int idx = i*imagePreproc.rows*imagePreproc.cols;
        int ch = netRT->input_dim.c-1 -i;
        memcpy((void*)&input[idx + netRT->input_dim.tot()*bi], (void*)bgr[ch].data, imagePreproc.rows*imagePreproc.cols*sizeof(dnnType));     
    }
    checkCuda(cudaMemcpyAsync(input_d + netRT->input_dim.tot()*bi, input + netRT->input_dim.tot()*bi, netRT->input_dim.tot()*sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));
#endif
}

void Yolo3Detection::postprocess(const int bi, const bool mAP){

    //get yolo outputs
    std::vector<float *> rt_out;
    //dnnType *rt_out[netRT->pluginFactory->n_yolos];
    for(int i=0; i<netRT->pluginFactory->n_yolos; i++)
        rt_out.push_back((dnnType*)netRT->buffersRT[i+1] + netRT->buffersDIM[i+1].tot()*bi);

    float x_ratio =  float(originalSize[bi].width) / float(netRT->input_dim.w);
    float y_ratio =  float(originalSize[bi].height) / float(netRT->input_dim.h);

    // compute dets
    nDets = 0;
    for(int i=0; i<netRT->pluginFactory->n_yolos; i++) {
        yolo[i]->dstData = rt_out[i];
        yolo[i]->computeDetections(dets, nDets, netRT->input_dim.w, netRT->input_dim.h, confThreshold, yolo[i]->new_coords);
    }
    tk::dnn::Yolo::mergeDetections(dets, nDets, classes, yolo[0]->nms_thresh, yolo[0]->nsm_kind);

    // fill detected
    detected.clear();
    for(int j=0; j<nDets; j++) {
        tk::dnn::Yolo::box b = dets[j].bbox;
        float x0   = (b.x-b.w/2.);
        float x1   = (b.x+b.w/2.);
        float y0   = (b.y-b.h/2.);
        float y1   = (b.y+b.h/2.);

        // convert to image coords
        x0 = x_ratio*x0;
        x1 = x_ratio*x1;
        y0 = y_ratio*y0;
        y1 = y_ratio*y1;
        
        for(int c=0; c<classes; c++) {
            if(dets[j].prob[c] >= confThreshold) {
                int obj_class = c;
                float prob = dets[j].prob[c];

                tk::dnn::box res;
                res.cl = obj_class;
                res.prob = prob;
                res.x = x0;
                res.y = y0;
                res.w = x1 - x0;
                res.h = y1 - y0;

                // FIXME: this shuld be useless
                // if(mAP)
                //     for(int c=0; c<classes; c++) 
                //         res.probs.push_back(dets[j].prob[c]);

                detected.push_back(res);
            }
        }

    }
    batchDetected.push_back(detected);
}


tk::dnn::Yolo* Yolo3Detection::getYoloLayer(int n) {
    if(n<3)
        return yolo[n];
    else 
        return nullptr;
}

}}
