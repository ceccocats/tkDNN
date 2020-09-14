#include "MobilenetDetection.h"

bool boxProbCmp(const tk::dnn::box &a, const tk::dnn::box &b){
    return (a.prob > b.prob);
}

namespace tk{ namespace dnn{

void MobilenetDetection::generate_ssd_priors(const SSDSpec *specs, const int n_specs, bool clamp){
    nPriors = 0;
    for (int i = 0; i < n_specs; i++){
        nPriors += specs[i].featureSize * specs[i].featureSize * 6;
    }

    priors = (float *)malloc(N_COORDS * nPriors * sizeof(float));

    int i_prio = 0;
    float scale, x_center, y_center, h, w, size, ratio;
    int min, max;
    for (int i = 0; i < n_specs; i++){
        scale = (float)imageSize / (float)specs[i].shrinkage;
        min = specs[i].boxHeight > specs[i].boxWidth ? specs[i].boxWidth : specs[i].boxHeight;
        max = specs[i].boxHeight < specs[i].boxWidth ? specs[i].boxWidth : specs[i].boxHeight;
        for (int j = 0; j < specs[i].featureSize; j++){
            for (int k = 0; k < specs[i].featureSize; k++){
                //small sized square box
                size = min;
                x_center = (k + 0.5f) / scale;
                y_center = (j + 0.5f) / scale;
                h = w = (float)size / (float)imageSize;

                priors[i_prio * N_COORDS + 0] = x_center;
                priors[i_prio * N_COORDS + 1] = y_center;
                priors[i_prio * N_COORDS + 2] = w;
                priors[i_prio * N_COORDS + 3] = h;
                ++i_prio;

                //big sized square box
                size = sqrt(max * min);
                h = w = (float)size / (float)imageSize;

                priors[i_prio * N_COORDS + 0] = x_center;
                priors[i_prio * N_COORDS + 1] = y_center;
                priors[i_prio * N_COORDS + 2] = w;
                priors[i_prio * N_COORDS + 3] = h;
                ++i_prio;

                //change h/w ratio of the small sized box
                size = min;
                h = w = size / (float)imageSize;
                ratio = sqrt(specs[i].ratio1);
                priors[i_prio * N_COORDS + 0] = x_center;
                priors[i_prio * N_COORDS + 1] = y_center;
                priors[i_prio * N_COORDS + 2] = w * ratio;
                priors[i_prio * N_COORDS + 3] = h / ratio;
                ++i_prio;

                priors[i_prio * N_COORDS + 0] = x_center;
                priors[i_prio * N_COORDS + 1] = y_center;
                priors[i_prio * N_COORDS + 2] = w / ratio;
                priors[i_prio * N_COORDS + 3] = h * ratio;
                ++i_prio;

                ratio = sqrt(specs[i].ratio2);
                priors[i_prio * N_COORDS + 0] = x_center;
                priors[i_prio * N_COORDS + 1] = y_center;
                priors[i_prio * N_COORDS + 2] = w * ratio;
                priors[i_prio * N_COORDS + 3] = h / ratio;
                ++i_prio;

                priors[i_prio * N_COORDS + 0] = x_center;
                priors[i_prio * N_COORDS + 1] = y_center;
                priors[i_prio * N_COORDS + 2] = w / ratio;
                priors[i_prio * N_COORDS + 3] = h * ratio;
                ++i_prio;
            }
        }
    }

    if (clamp){
        for (int i = 0; i < nPriors * N_COORDS; i++){
            priors[i] = priors[i] > 1.0f ? 1.0f : priors[i];
            priors[i] = priors[i] < 0.0f ? 0.0f : priors[i];
        }
    }
}

void MobilenetDetection::convert_locatios_to_boxes_and_center(){
    float cur_x, cur_y;
    for (int i = 0; i < nPriors; i++){
        locations_h[i * N_COORDS + 0] = locations_h[i * N_COORDS + 0] * centerVariance * priors[i * N_COORDS + 2] + priors[i * N_COORDS + 0];
        locations_h[i * N_COORDS + 1] = locations_h[i * N_COORDS + 1] * centerVariance * priors[i * N_COORDS + 3] + priors[i * N_COORDS + 1];
        locations_h[i * N_COORDS + 2] = exp(locations_h[i * N_COORDS + 2] * sizeVariance) * priors[i * N_COORDS + 2];
        locations_h[i * N_COORDS + 3] = exp(locations_h[i * N_COORDS + 3] * sizeVariance) * priors[i * N_COORDS + 3];

        cur_x = locations_h[i * N_COORDS + 0];
        cur_y = locations_h[i * N_COORDS + 1];

        locations_h[i * N_COORDS + 0] = cur_x - locations_h[i * N_COORDS + 2] / 2;
        locations_h[i * N_COORDS + 1] = cur_y - locations_h[i * N_COORDS + 3] / 2;
        locations_h[i * N_COORDS + 2] = cur_x + locations_h[i * N_COORDS + 2] / 2;
        locations_h[i * N_COORDS + 3] = cur_y + locations_h[i * N_COORDS + 3] / 2;
    }
}

float MobilenetDetection::iou(const tk::dnn::box &a, const tk::dnn::box &b){
    float max_x = a.x > b.x ? a.x : b.x;
    float max_y = a.y > b.y ? a.y : b.y;
    float min_w = a.w < b.w ? a.w : b.w;
    float min_h = a.h < b.h ? a.h : b.h;

    float ao_w = min_w - max_x > 0 ? min_w - max_x : 0;
    float ao_h = min_h - max_y > 0 ? min_h - max_y : 0;

    float area_overlap = ao_w * ao_h;
    float area_0_w = a.w - a.x > 0 ? a.w - a.x : 0;
    float area_0_h = a.h - a.y > 0 ? a.h - a.y : 0;

    float area_1_w = b.w - b.x > 0 ? b.w - b.x : 0;
    float area_1_h = b.h - b.y > 0 ? b.h - b.y : 0;

    float area_0 = area_0_h * area_0_w;
    float area_1 = area_1_h * area_1_w;

    float iou = area_overlap / (area_0 + area_1 - area_overlap + 1e-5);
    return iou;
}

bool MobilenetDetection::init(const std::string& tensor_path, const int n_classes, const int n_batches, const float conf_thresh){
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str());
    imageSize = netRT->input_dim.h;
    classes = n_classes;
    nBatches = n_batches;
    confThreshold = conf_thresh;

    SSDSpec specs[N_SSDSPEC];

    if(imageSize == 300){
        specs[0].setAll(19, 16, 60, 105, 2, 3);
        specs[1].setAll(10, 32, 105, 150, 2, 3);
        specs[2].setAll(5, 64, 150, 195, 2, 3);
        specs[3].setAll(3, 100, 195, 240, 2, 3);
        specs[4].setAll(2, 150, 240, 285, 2, 3);
        specs[5].setAll(1, 300, 285, 330, 2, 3);
    }
    else if(imageSize == 512){
        specs[0].setAll(32, 16, 60, 105, 2, 3);
        specs[1].setAll(16, 32, 105, 150, 2, 3);
        specs[2].setAll(8, 64, 150, 195, 2, 3);
        specs[3].setAll(4, 100, 195, 240, 2, 3);
        specs[4].setAll(2, 150, 240, 285, 2, 3);
        specs[5].setAll(1, 300, 285, 330, 2, 3);
    }  
    else{
        FatalError("Input size for mobilenet not supported");
    }

    generate_ssd_priors(specs, N_SSDSPEC);

#ifndef OPENCV_CUDACONTRIB
    checkCuda(cudaMallocHost(&input, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));
#endif
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType) * netRT->input_dim.tot() * nBatches));

    locations_h = (float *)malloc(N_COORDS * nPriors * sizeof(float));
    confidences_h = (float *)malloc(nPriors * classes * sizeof(float));

    for (int c = 0; c < classes; c++){
        int offset = c * 123457 % classes;
        float r = getColor(2, offset, classes);
        float g = getColor(1, offset, classes);
        float b = getColor(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0 * b), int(255.0 * g), int(255.0 * r));
    }

    if(classes == 11){ //BDD
        const char *classes_names_[] = {
        "person","car","truck","bus","motor","bike","rider","traffic light","traffic sign","train"};
        classesNames = std::vector<std::string>(classes_names_, std::end(classes_names_));
    }
    else if(classes == 21){ //VOC
        const char *classes_names_[] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
        classesNames = std::vector<std::string>(classes_names_, std::end(classes_names_));

    }
    else if (classes == 81){ //COCO
        const char *classes_names_[] = {
        "person" , "bicycle" , "car" , "motorbike" , "aeroplane" , "bus" ,
        "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "stop sign" , 
        "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , 
        "elephant" , "bear" , "zebra" , "giraffe" , "backpack" , "umbrella" , "handbag" , 
        "tie" , "suitcase" , "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , 
        "baseball bat" , "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , 
        "bottle" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , "banana" , 
        "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" , "pizza" , 
        "donut" , "cake" , "chair" , "sofa" , "pottedplant" , "bed" , "diningtable" , 
        "toilet" , "tvmonitor" , "laptop" , "mouse" , "remote" , "keyboard" , 
        "cell phone" , "microwave" , "oven" , "toaster" , "sink" , "refrigerator" , 
        "book" , "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush"};
        classesNames = std::vector<std::string>(classes_names_, std::end(classes_names_));

    }
    else{
        FatalError("Number of classes not supported for mobilenet");
    }
    return 1;
}

void MobilenetDetection::preprocess(cv::Mat &frame, const int bi){
#ifdef OPENCV_CUDACONTRIB
        //move original image on GPU
        cv::cuda::GpuMat orig_img, frame_nomean;
        orig_img = cv::cuda::GpuMat(frame);

        //resize image, remove mean, divide by std
        cv::cuda::resize (orig_img, orig_img, cv::Size(netRT->input_dim.w, netRT->input_dim.h)); 
        orig_img.convertTo(frame_nomean, CV_32FC3, 1, -127);
        frame_nomean.convertTo(imagePreproc, CV_32FC3, 1 / 128.0, 0);

        //copy image into tensors
        cv::cuda::split(imagePreproc, bgr);

        for(int i=0; i < netRT->input_dim.c; i++){
            int idx = i * imagePreproc.rows * imagePreproc.cols;
            checkCuda( cudaMemcpy((void *)&input_d[idx + netRT->input_dim.tot()*bi], (void *)bgr[i].data, imagePreproc.rows * imagePreproc.cols* sizeof(float), cudaMemcpyDeviceToDevice) );
        }
#else
        //resize image, remove mean, divide by std
        cv::Mat frame_nomean;
        resize(frame, frame, cv::Size(netRT->input_dim.w, netRT->input_dim.h));
        frame.convertTo(frame_nomean, CV_32FC3, 1, -127);
        frame_nomean.convertTo(imagePreproc, CV_32FC3, 1 / 128.0, 0);

        //copy image into tensor and copy it into GPU
        cv::split(imagePreproc, bgr);
        for (int i = 0; i < netRT->input_dim.c; i++){
            int idx = i * imagePreproc.rows * imagePreproc.cols;
            memcpy((void *)&input[idx + netRT->input_dim.tot()*bi], (void *)bgr[i].data, imagePreproc.rows * imagePreproc.cols * sizeof(dnnType));
        }
        checkCuda(cudaMemcpyAsync(input_d+ netRT->input_dim.tot()*bi, input + netRT->input_dim.tot()*bi, netRT->input_dim.tot() * sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));
#endif
}

void MobilenetDetection::postprocess(const int bi, const bool mAP){
    //get confidences and locations_h
    dnnType *rt_out[2];
    rt_out[0] = (dnnType *)netRT->buffersRT[3]+ netRT->buffersDIM[3].tot()*bi;
    rt_out[1] = (dnnType *)netRT->buffersRT[4]+ netRT->buffersDIM[4].tot()*bi;

    detected.clear();

    checkCuda(cudaMemcpy(confidences_h, rt_out[0], nPriors * classes * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(locations_h, rt_out[1], N_COORDS * nPriors * sizeof(float), cudaMemcpyDeviceToHost));
    convert_locatios_to_boxes_and_center();

    int width =  originalSize[bi].width;
    int height =  originalSize[bi].height;

    float *conf_per_class;
    for (int i = 1; i < classes; i++){
        conf_per_class = &confidences_h[i * nPriors];
        std::vector<tk::dnn::box> boxes;
        for (int j = 0; j < nPriors; j++){

            if (conf_per_class[j] > confThreshold){
                tk::dnn::box b;
                b.cl = i;
                b.prob = conf_per_class[j];
                b.x = locations_h[j * N_COORDS + 0];
                b.y = locations_h[j * N_COORDS + 1];
                b.w = locations_h[j * N_COORDS + 2];
                b.h = locations_h[j * N_COORDS + 3];

                if(mAP)
                    for(int c=1; c<classes; c++) 
                        b.probs.push_back(confidences_h[c * nPriors + j]);

                boxes.push_back(b);
            }
        }
        std::sort(boxes.begin(), boxes.end(), boxProbCmp);

        std::vector<tk::dnn::box> remaining;        
        while (boxes.size() > 0){
            remaining.clear();

            tk::dnn::box b;
            b.cl = boxes[0].cl -1 ;             //remove background class
            b.prob = boxes[0].prob;
            b.x = boxes[0].x * width; 
            b.y = boxes[0].y * height;
            b.w = boxes[0].w * width - b.x;     //convert from x1 to width
            b.h = boxes[0].h * height - b.y;    //convert from y1 to height
            detected.push_back(b);
            for (size_t j = 1; j < boxes.size(); j++){
                if (iou(boxes[0], boxes[j]) <= IoUThreshold){
                    remaining.push_back(boxes[j]);
                }
            }
            boxes = remaining;
        }
    }
    batchDetected.push_back(detected);
}


} // namespace dnn
} // namespace tk