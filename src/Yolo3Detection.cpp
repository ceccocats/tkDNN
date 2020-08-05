#include "Yolo3Detection.h"


namespace tk { namespace dnn {

bool Yolo3Detection::init(const std::string& tensor_path, const int n_classes, const int n_batches) {

    //convert network to tensorRT
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );

    nBatches = n_batches;
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

cv::Mat resize_image(cv::Mat im, int w, int h)
{
    cv::Mat resized = cv::Mat(cv::Size(w,h), CV_32FC3, cv::Scalar(0) );
    cv::Mat part = cv::Mat(cv::Size(w,im.rows), CV_32FC3, cv::Scalar(0) );
    int r, c, k;
    float w_scale = (float)(im.cols - 1) / (w - 1);
    float h_scale = (float)(im.rows - 1) / (h - 1);
    
    for(k = 0; k < im.channels(); ++k){
        for(r = 0; r < im.rows; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.cols == 1){
                    val = im.at<cv::Vec3f>(r, im.cols-1)[k];
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * im.at<cv::Vec3f>(r, ix)[k] + dx * im.at<cv::Vec3f>(r,ix+1)[k];
                }
                part.at<cv::Vec3f>(r,c)[k] = val;
            }
        }
    }
    for(k = 0; k < im.channels(); ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * part.at<cv::Vec3f>(iy, c)[k];
                resized.at<cv::Vec3f>(r, c)[k] = val;
            }
            if(r == h-1 || im.rows == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * part.at<cv::Vec3f>(iy+1, c)[k];
                resized.at<cv::Vec3f>(r,c)[k] += val;
            }
        }
    }

    return resized;
}

void Yolo3Detection::preprocess(cv::Mat &frame, const int bi){
    frame.convertTo(imagePreproc, CV_32FC3, 1/255.0); 

    if(letterbox){
        int im_w = frame.cols;
        int im_h = frame.rows;
        int net_w = netRT->input_dim.w;
        int net_h = netRT->input_dim.h;
        if(net_w == net_h && letterbox){
            float ratio = ( im_w > im_h ) ? float(im_w)/float(net_w) : float(im_h)/float(net_h);

            int new_h = im_h/ratio;
            int new_w = im_w/ratio;

            imagePreproc = resize_image(imagePreproc, new_w, new_h);

            cv::Mat borders;
            int top = (net_h - new_h)/2;
            int bottom = (net_h - new_h) - top;
            int left = (net_w - new_w)/2;
            int right = (net_w - new_w) - left;

            cv::copyMakeBorder(imagePreproc,imagePreproc, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0.5,0.5,0.5));
        }
        else
            FatalError("letterbox not spported with h!=w");
    }
    else
        imagePreproc = resize_image(imagePreproc, netRT->input_dim.w, netRT->input_dim.h);

    //split channels
    cv::split(imagePreproc,bgr);//split source

    //write channels
    for(int i=0; i<netRT->input_dim.c; i++) {
        int idx = i*imagePreproc.rows*imagePreproc.cols;
        int ch = netRT->input_dim.c-1 -i;
        memcpy((void*)&input[idx + netRT->input_dim.tot()*bi], (void*)bgr[ch].data, imagePreproc.rows*imagePreproc.cols*sizeof(dnnType));     
    }
    checkCuda(cudaMemcpyAsync(input_d + netRT->input_dim.tot()*bi, input + netRT->input_dim.tot()*bi, netRT->input_dim.tot()*sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));
}

void Yolo3Detection::postprocess(const int bi, const bool mAP){

    //get yolo outputs
    dnnType *rt_out[netRT->pluginFactory->n_yolos]; 
    for(int i=0; i<netRT->pluginFactory->n_yolos; i++) 
        rt_out[i] = (dnnType*)netRT->buffersRT[i+1] + netRT->buffersDIM[i+1].tot()*bi;

    float x_ratio =  float(originalSize[bi].width) / float(netRT->input_dim.w);
    float y_ratio =  float(originalSize[bi].height) / float(netRT->input_dim.h);

    // compute dets
    nDets = 0;
    for(int i=0; i<netRT->pluginFactory->n_yolos; i++) {
        yolo[i]->dstData = rt_out[i];
        yolo[i]->computeDetections(dets, nDets, netRT->input_dim.w, netRT->input_dim.h, confThreshold);
    }
    tk::dnn::Yolo::mergeDetections(dets, nDets, classes);

    int im_w = originalSize[bi].width;
    int im_h = originalSize[bi].height;
    int net_w = netRT->input_dim.w;
    int net_h = netRT->input_dim.h;
    int new_h, new_w;

    int top = 0, left = 0;

    if(letterbox){
        float ratio = ( im_w > im_h ) ? float(im_w)/float(net_w) : float(im_h)/float(net_h);
        x_ratio = ratio;
        y_ratio = ratio;
        std::cout<<ratio<<std::endl;

        int new_h = im_h/ratio;
        int new_w = im_w/ratio;

        top = (net_h - new_h)/2;
        left = (net_w - new_w)/2;
    }
    else{
        new_h = net_h;
        new_w = net_w;
    }
    
    float deltaw = net_w - new_w;
    float deltah = net_h - new_h;
    float ratiow = (float)new_w / net_w;
    float ratioh = (float)new_h / net_h;

    // fill detected
    detected.clear();
    for(int j=0; j<nDets; j++) {
        tk::dnn::Yolo::box b = dets[j].bbox;

        float x0   = (b.x - left - b.w/2.);
        float x1   = (b.x - left + b.w/2.);
        float y0   = (b.y - top - b.h/2.);
        float y1   = (b.y - top + b.h/2.);

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
