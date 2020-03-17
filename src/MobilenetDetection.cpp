#include "MobilenetDetection.h"

bool boxProbCmp(const tk::dnn::box &a, const tk::dnn::box &b)
{
    return (a.prob > b.prob);
}

namespace tk
{
namespace dnn
{

void MobilenetDetection::generate_ssd_priors(const SSDSpec *specs, const int n_specs, bool clamp)
{
    n_priors = 0;
    for (int i = 0; i < n_specs; i++)
    {
        n_priors += specs[i].featureSize * specs[i].featureSize * 6;
    }

    // std::cout<<"n priors: "<<n_priors<<std::endl;
    // std::cout<<"n priors: "<<n_specs<<std::endl;

    priors = (float *)malloc(N_COORDS * n_priors * sizeof(float));

    int i_prio = 0;
    float scale, x_center, y_center, h, w, size, ratio;
    int min, max;
    for (int i = 0; i < n_specs; i++)
    {
        scale = (float)imageSize / (float)specs[i].shrinkage;
        min = specs[i].boxHeight > specs[i].boxWidth ? specs[i].boxWidth : specs[i].boxHeight;
        max = specs[i].boxHeight < specs[i].boxWidth ? specs[i].boxWidth : specs[i].boxHeight;
        for (int j = 0; j < specs[i].featureSize; j++)
        {
            for (int k = 0; k < specs[i].featureSize; k++)
            {
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

    if (clamp)
    {
        for (int i = 0; i < n_priors * N_COORDS; i++)
        {
            priors[i] = priors[i] > 1.0f ? 1.0f : priors[i];
            priors[i] = priors[i] < 0.0f ? 0.0f : priors[i];

            // std::cout<<priors[i]<<" ";
            // if((i+1)%4 == 0)
            //     std::cout<< i/4 <<" " <<std::endl;
        }
    }
}

void MobilenetDetection::convert_locatios_to_boxes_and_center(float *priors, const int n_priors, float *locations, const float centerVariance, const float sizeVariance)
{
    float cur_x, cur_y;
    for (int i = 0; i < n_priors; i++)
    {
        locations[i * N_COORDS + 0] = locations[i * N_COORDS + 0] * centerVariance * priors[i * N_COORDS + 2] + priors[i * N_COORDS + 0];
        locations[i * N_COORDS + 1] = locations[i * N_COORDS + 1] * centerVariance * priors[i * N_COORDS + 3] + priors[i * N_COORDS + 1];
        locations[i * N_COORDS + 2] = exp(locations[i * N_COORDS + 2] * sizeVariance) * priors[i * N_COORDS + 2];
        locations[i * N_COORDS + 3] = exp(locations[i * N_COORDS + 3] * sizeVariance) * priors[i * N_COORDS + 3];

        cur_x = locations[i * N_COORDS + 0];
        cur_y = locations[i * N_COORDS + 1];

        locations[i * N_COORDS + 0] = cur_x - locations[i * N_COORDS + 2] / 2;
        locations[i * N_COORDS + 1] = cur_y - locations[i * N_COORDS + 3] / 2;
        locations[i * N_COORDS + 2] = cur_x + locations[i * N_COORDS + 2] / 2;
        locations[i * N_COORDS + 3] = cur_y + locations[i * N_COORDS + 3] / 2;

        // std::cout<<locations[i*N_COORDS + 0]<<" "<<locations[i*N_COORDS + 1]<<" "<<locations[i*N_COORDS + 2]<<" "<<locations[i*N_COORDS + 3]<<" "<<std::endl;
    }
}

float MobilenetDetection::iou(const tk::dnn::box &a, const tk::dnn::box &b)
{
    float max_x = a.x > b.x ? a.x : b.x;
    float max_y = a.y > b.y ? a.y : b.y;
    float min_w = a.w < b.w ? a.w : b.w;
    float min_h = a.h < b.h ? a.h : b.h;

    float ao_w = min_w - max_x > 0 ? min_w - max_x : 0;
    float ao_h = min_h - max_y > 0 ? min_h - max_y : 0;

    // std::cout<<" ao w: "<<ao_w<<" ao h: "<<ao_h<<std::endl;

    float area_overlap = ao_w * ao_h;
    float area_0_w = a.w - a.x > 0 ? a.w - a.x : 0;
    float area_0_h = a.h - a.y > 0 ? a.h - a.y : 0;

    float area_1_w = b.w - b.x > 0 ? b.w - b.x : 0;
    float area_1_h = b.h - b.y > 0 ? b.h - b.y : 0;

    float area_0 = area_0_h * area_0_w;
    float area_1 = area_1_h * area_1_w;

    // std::cout<<" area_overlap : "<<area_overlap<<" area_0: "<<area_0<<" area_1: "<<area_1<<std::endl;

    float iou = area_overlap / (area_0 + area_1 - area_overlap + 1e-5);
    return iou;
}

std::vector<tk::dnn::box> MobilenetDetection::postprocess(float *locations, float *confidences, const int n_values, const float threshold, const int n_classes, const float iou_thresh, const int width, const int height)
{
    float *conf_per_class;
    std::vector<tk::dnn::box> detections;
    for (int i = 1; i < n_classes; i++)
    {
        conf_per_class = &confidences[i * n_values];
        std::vector<tk::dnn::box> boxes;
        for (int j = 0; j < n_values; j++)
        {

            if (conf_per_class[j] > threshold)
            {
                tk::dnn::box b;
                b.cl = i;
                b.prob = conf_per_class[j];
                b.x = locations[j * N_COORDS + 0];
                b.y = locations[j * N_COORDS + 1];
                b.w = locations[j * N_COORDS + 2];
                b.h = locations[j * N_COORDS + 3];

                boxes.push_back(b);
            }
        }
        std::sort(boxes.begin(), boxes.end(), boxProbCmp);
        // for(auto b:boxes)
        //     b.print();
        std::vector<tk::dnn::box> remaining;
        while (boxes.size() > 0)
        {
            remaining.clear();

            tk::dnn::box b;
            b.cl = boxes[0].cl;
            b.prob = boxes[0].prob;
            b.x = boxes[0].x * width;
            b.y = boxes[0].y * height;
            b.w = boxes[0].w * width;
            b.h = boxes[0].h * height;
            detections.push_back(b);
            for (size_t j = 1; j < boxes.size(); j++)
            {
                if (iou(boxes[0], boxes[j]) <= iou_thresh)
                {
                    remaining.push_back(boxes[j]);
                }
            }
            boxes = remaining;
        }
    }
    // std::cout<<"picked"<<std::endl;
    // for(auto b:detections)
    //     b.print();

    return detections;
}

float MobilenetDetection::get_color2(int c, int x, int max)
{
    float ratio = ((float)x / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * __colors[i % 6][c % 3] + ratio * __colors[j % 6][c % 3];
    //printf("%f\n", r);
    return r;
}

void MobilenetDetection::init(std::string tensor_path, int input_size, int n_classes)
{
    this->imageSize = input_size;
    this->classes = n_classes;

    const int n_SSDSpec = 6;
    SSDSpec specs[6];


    if(input_size == 300)
    {
        specs[0].setAll(19, 16, 60, 105, 2, 3);
        specs[1].setAll(10, 32, 105, 150, 2, 3);
        specs[2].setAll(5, 64, 150, 195, 2, 3);
        specs[3].setAll(3, 100, 195, 240, 2, 3);
        specs[4].setAll(2, 150, 240, 285, 2, 3);
        specs[5].setAll(1, 300, 285, 330, 2, 3);
    }
    else if(input_size == 512)
    {
        specs[0].setAll(32, 16, 60, 105, 2, 3);
        specs[1].setAll(16, 32, 105, 150, 2, 3);
        specs[2].setAll(8, 64, 150, 195, 2, 3);
        specs[3].setAll(4, 100, 195, 240, 2, 3);
        specs[4].setAll(2, 150, 240, 285, 2, 3);
        specs[5].setAll(1, 300, 285, 330, 2, 3);
    }  
    else
    {
        FatalError("Input size for mobilenet not supported");
    }
     

    generate_ssd_priors(specs, n_SSDSpec);

    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str());

    checkCuda(cudaMallocHost(&input, sizeof(dnnType) * netRT->input_dim.tot()));
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType) * netRT->input_dim.tot()));

    locations_h = (float *)malloc(N_COORDS * n_priors * sizeof(float));
    confidences_h = (float *)malloc(n_priors * classes * sizeof(float));

    dim = tk::dnn::dataDim_t(1, 3, imageSize, imageSize, 1);

    for (int c = 0; c < classes; c++)
    {
        int offset = c * 123457 % classes;
        float r = get_color2(2, offset, classes);
        float g = get_color2(1, offset, classes);
        float b = get_color2(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0 * b), int(255.0 * g), int(255.0 * r));
    }

    if(classes == 21)
    {
        const char *classes_names_[] = {
        "BACKGROUND", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
        classesNames = std::vector<std::string>(classes_names_, std::end(classes_names_));

    }
    else if (classes == 81)
    {
        const char *classes_names_[] = {
        "BACKGROUND", "person" , "bicycle" , "car" , "motorbike" , "aeroplane" , "bus" ,
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
    else
    {
        FatalError("Number of classes not supported for mobilenet");
    }
    
    
}

cv::Mat MobilenetDetection::draw()
{
    tk::dnn::box b;
    for (size_t i = 0; i < detected.size(); i++)
    {
        b = detected[i];
        std::string det_class = classesNames[b.cl];
        cv::rectangle(origImg, cv::Point(b.x, b.y), cv::Point(b.w, b.h), colors[b.cl], 2);
        // draw label
        cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::rectangle(origImg, cv::Point(b.x, b.y), cv::Point((b.x + textSize.width - 2), (b.y - textSize.height - 2)), colors[b.cl], -1);
        cv::putText(origImg, det_class, cv::Point(b.x, (b.y - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
    }
    return origImg;
}

void MobilenetDetection::update(cv::Mat &img)
{
    TIMER_START
    detected.clear();

    //save origin image
    origImg = img;
    cv::Size sz = origImg.size();

    //resize image, remove mean, divide by std
    cv::Mat frame_resize, frame_nomean, frame_scaled;
    resize(origImg, frame_resize, cv::Size(netRT->input_dim.w, netRT->input_dim.h));
    frame_resize.convertTo(frame_nomean, CV_32FC3, 1, -127);
    frame_nomean.convertTo(frame_scaled, CV_32FC3, 1 / 128.0, 0);

    //copy image into tensor and copy it into GPU
    cv::split(frame_scaled, bgr);
    for (int i = 0; i < netRT->input_dim.c; i++)
    {
        int idx = i * frame_scaled.rows * frame_scaled.cols;
        memcpy((void *)&input[idx], (void *)bgr[i].data, frame_scaled.rows * frame_scaled.cols * sizeof(dnnType));
    }
    checkCuda(cudaMemcpyAsync(input_d, input, netRT->input_dim.tot() * sizeof(dnnType), cudaMemcpyHostToDevice, netRT->stream));

    //do inference
    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TIMER_START
        netRT->infer(dim2, input_d);
        TIMER_STOP
        dim2.print();
    }

    //get confidences and locations
    conf = (dnnType *)netRT->buffersRT[3];
    loc = (dnnType *)netRT->buffersRT[4];

    checkCuda(cudaMemcpy(confidences_h, conf, n_priors * classes * sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(locations_h, loc, N_COORDS * n_priors * sizeof(float), cudaMemcpyDeviceToHost));

    //postprocess
    convert_locatios_to_boxes_and_center(priors, n_priors, locations_h, centerVariance, sizeVariance);
    detected = postprocess(locations_h, confidences_h, n_priors, confThresh, classes, iouThreshold, sz.width, sz.height);

    TIMER_STOP
    stats.push_back(t_ns);
}

} // namespace dnn
} // namespace tk