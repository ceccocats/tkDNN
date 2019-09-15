#include "Yolo3Detection.h"

namespace tk { namespace dnn {

float _colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * _colors[i % 6][c % 3] + ratio*_colors[j % 6][c % 3];
    //printf("%f\n", r);
    return r;
}

bool Yolo3Detection::init(std::string tensor_path) {

    //const char *tensor_path = "../data/yolo3/yolo3_berkeley.rt";

    //convert network to tensorRT
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );

    if(netRT->pluginFactory->n_yolos != 3) {
        FatalError("this is not yolo3");
    }

    for(int i=0; i<netRT->pluginFactory->n_yolos; i++) {
        YoloRT *yRT = netRT->pluginFactory->yolos[i];
        classes = yRT->classes;
        num = yRT->num;

        // make a yolo layer for interpret predictions
        yolo[i] = new tk::dnn::Yolo(nullptr, classes, num, ""); // yolo without input and bias
        yolo[i]->mask_h = new dnnType[num];
        yolo[i]->bias_h = new dnnType[num*3*2];
        memcpy(yolo[i]->mask_h, yRT->mask, sizeof(dnnType)*num);
        memcpy(yolo[i]->bias_h, yRT->bias, sizeof(dnnType)*num*3*2);
        yolo[i]->input_dim = yolo[i]->output_dim = tk::dnn::dataDim_t(1, yRT->c, yRT->h, yRT->w);
    }

    dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);
    
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*netRT->input_dim.tot()));
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*netRT->input_dim.tot()));


    // class colors precompute    
    for(int c=0; c<classes; c++) {
        int offset = c*123457 % classes;
        float r = get_color(2, offset, classes);
        float g = get_color(1, offset, classes);
        float b = get_color(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }
    return true;
} 


void Yolo3Detection::update(cv::Mat &imageORIG) {

    if(!imageORIG.data) {
        std::cout<<"YOLO: NO IMAGE DATA\n";
        return;
    }     
    float xRatio =  float(imageORIG.cols) / float(netRT->input_dim.w);
    float yRatio =  float(imageORIG.rows) / float(netRT->input_dim.h);

    resize(imageORIG, imageORIG, cv::Size(netRT->input_dim.w, netRT->input_dim.h));

    imageORIG.convertTo(imageF, CV_32FC3, 1/255.0); 

    //split channels
    cv::split(imageF,bgr);//split source

    //write channels
    for(int i=0; i<netRT->input_dim.c; i++) {
        int idx = i*imageF.rows*imageF.cols;
        int ch = netRT->input_dim.c-1 -i;
        memcpy((void*)&input[idx], (void*)bgr[ch].data, imageF.rows*imageF.cols*sizeof(dnnType));
    }


    //DO INFERENCE
    dnnType *rt_out[3]; 
    tk::dnn::dataDim_t dim = netRT->input_dim;
    checkCuda(cudaMemcpyAsync(input_d, input, dim.tot()*sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));

    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim.print();
        TIMER_START
        netRT->infer(dim, input_d);
        TIMER_STOP
        dim.print();
    }
    
    TIMER_START
    // compute dets
    ndets = 0;
    for(int i=0; i<3; i++) {
        rt_out[i] = (dnnType*)netRT->buffersRT[i+1];
        yolo[i]->dstData = rt_out[i];
        yolo[i]->computeDetections(dets, ndets, netRT->input_dim.w, netRT->input_dim.h, thresh);
    }
    tk::dnn::Yolo::mergeDetections(dets, ndets, classes);
    TIMER_STOP

    // fill detected
    detected.clear();
    for(int j=0; j<ndets; j++) {
        tk::dnn::Yolo::box b = dets[j].bbox;
        int x0   = (b.x-b.w/2.);
        int x1   = (b.x+b.w/2.);
        int y0   = (b.y-b.h/2.);
        int y1   = (b.y+b.h/2.);
        int obj_class = -1;
        float prob = 0;
        for(int c=0; c<classes; c++) {
            if(dets[j].prob[c] >= thresh) {
                obj_class = c;
                prob = dets[j].prob[c];
            }
        }

        if(obj_class >= 0) {
            //std::cout<<obj_class<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
            //cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1), colors[obj_class], 2);
            
            // convert to image coords
            x0 = xRatio*x0;
            x1 = xRatio*x1;
            y0 = yRatio*y0;
            y1 = yRatio*y1;
              
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

}}
