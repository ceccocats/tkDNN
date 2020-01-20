#include "CenternetDetection.h"

namespace tk { namespace dnn {

float __colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
float get_color2(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * __colors[i % 6][c % 3] + ratio*__colors[j % 6][c % 3];
    //printf("%f\n", r);
    return r;
}

bool CenternetDetection::init(std::string tensor_path) {
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );
    
    dim = tk::dnn::dataDim_t(1, 3, 224, 224, 1);
    const char *coco_class_name_[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", 
            "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
            };
    coco_class_name = std::vector<std::string>(coco_class_name_, std::end( coco_class_name_ ));
    src = cv::Mat(cv::Size(2,3), CV_32F);
    dst = cv::Mat(cv::Size(2,3), CV_32F);
    // dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);

    checkCuda(cudaMallocHost(&input_h, sizeof(dnnType)*netRT->input_dim.tot()));
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*netRT->input_dim.tot()));
    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*netRT->input_dim.tot()));

    dim_hm = tk::dnn::dataDim_t(1, 80, 56, 56, 1);
    dim_wh = tk::dnn::dataDim_t(1, 2, 56, 56, 1);
    dim_reg = tk::dnn::dataDim_t(1, 2, 56, 56, 1);

    checkCuda( cudaMalloc(&topk_scores, dim_hm.c * K *sizeof(float)) );
    checkCuda( cudaMalloc(&topk_inds_, dim_hm.c * K *sizeof(int)) );      
    checkCuda( cudaMalloc(&topk_ys_, dim_hm.c * K *sizeof(float)) );      
    checkCuda( cudaMalloc(&topk_xs_, dim_hm.c * K *sizeof(float)) );    
    checkCuda( cudaMalloc(&ids_d, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int)) );
    checkCuda( cudaMalloc(&ids_2d, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int)) );
    checkCuda( cudaMallocHost(&ids_, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int)) );
    checkCuda( cudaMallocHost(&ids_2, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int)) );
    for(int i =0; i<dim_hm.c * dim_hm.h * dim_hm.w; i++){
        ids_[i] = i;
    }
    int val = 0;
    for(int i =0; i <dim_hm.c *  dim_hm.h * dim_hm.w; i++){
        ids_2[i] = val;
        if(i%dim_hm.c == 0)
            val = 0;
    }

    checkCuda( cudaMallocHost(&scores, K *sizeof(float)) );
    checkCuda( cudaMalloc(&scores_d, K *sizeof(float)) );
    checkCuda( cudaMallocHost(&clses, K *sizeof(int)) );
    checkCuda( cudaMalloc(&clses_d, K *sizeof(int)) );
    // checkCuda( cudaMallocHost(&topk_inds, K *sizeof(int)) );
    checkCuda( cudaMalloc(&topk_inds_d, K *sizeof(int)) );
    checkCuda( cudaMalloc(&topk_ys_d, K *sizeof(float)) );     
    checkCuda( cudaMalloc(&topk_xs_d, K *sizeof(float)) ); 
    // checkCuda( cudaMalloc(&intid, K *sizeof(int)) );
    checkCuda( cudaMalloc(&inttopk_ys_d, K *sizeof(int)) );
    checkCuda( cudaMalloc(&inttopk_xs_d, K *sizeof(int)) );
    
    // checkCuda( cudaMalloc(&ids_d, dim_hm.c * K*sizeof(int)) );

    // checkCuda( cudaMallocHost(&wh_aus, dim_wh.tot()*sizeof(dnnType)) ); 
    checkCuda( cudaMallocHost(&bbx0, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bby0, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bbx1, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bby1, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bbx0_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bby0_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bbx1_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bby1_d, K * sizeof(float)) ); 

    checkCuda( cudaMallocHost(&target_coords, 4 * K *sizeof(float)) );

    mean << 0.408, 0.447, 0.47;
    stddev << 0.289, 0.274, 0.278;
}

void CenternetDetection::testdog() {

    readBinaryFile(input_bin, dim.tot(), &input_h, &input_d);

    // -------- transofrm compose
    cv::Mat imageORIG = cv::imread("../../dog.jpg");
    imageORIG.convertTo(imageF, CV_32FC3, 1/255.0); 
    sz = imageF.size();
    std::cout<<"image: "<<sz.width<<", "<<sz.height<<std::endl;
    resize(imageF, imageF, cv::Size(256, 256));
    const int cropSize = 224;
    const int offsetW = (imageF.cols - cropSize) / 2;
    const int offsetH = (imageF.rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
    imageF = imageF(roi).clone();
    std::cout << "Cropped image dimension: " << imageF.cols << " X " << imageF.rows << std::endl;

    mean << 0.485, 0.456, 0.406;
    stddev << 0.229, 0.224, 0.225;
    sz = imageF.size();
    // std::cout<<"size: "<<sz.height<<" "<<sz.width<<" - "<<std::endl;
    // std::cout<<"mean: "<<mean<<", std: "<<stddev<<std::endl;
    cv::add(imageF, -mean, imageF);
    cv::divide(imageF, stddev, imageF);
    //split channels
    cv::split(imageF,bgr);//split source
    dim2 = dim;
    //write channels
    for(int i=0; i<dim2.c; i++) {
        int idx = i*imageF.rows*imageF.cols;
        int ch = dim2.c-1 -i;
        memcpy((void*)&input[idx], (void*)bgr[ch].data, imageF.rows*imageF.cols*sizeof(dnnType));
    }

    checkCuda(cudaMemcpyAsync(input_d, input, dim2.tot()*sizeof(dnnType), cudaMemcpyHostToDevice));

    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        netRT->infer(dim2, input_d);
        TIMER_STOP
        dim2.print();
    }
    // checkResult(dim2.tot(), input_h, input);    
}

cv::Mat CenternetDetection::draw(cv::Mat &imageORIG) {

    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;
    int baseline = 0;
    float fontScale = 0.5;
    int thickness = 2;   

    for(int c=0; c<classes; c++) {
        int offset = c*123457 % classes;
        float r = get_color2(2, offset, classes);
        float g = get_color2(1, offset, classes);
        float b = get_color2(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }
    int num_detected = detected.size();
    for (int i = 0; i < num_detected; i++){
        b = detected[i];
        x0 = b.x;
        w = b.w;
        x1 = b.x + w;
        y0 = b.y;
        h = b.h;
        y1 = b.y + h;
        objClass = b.cl;
        det_class = coco_class_name[objClass];
        cv::rectangle(imageORIG, cv::Point(x0, y0), cv::Point(x1, y1), colors[objClass], 2);
        // draw label
        cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        cv::rectangle(imageORIG, cv::Point(x0, y0), cv::Point((x0 + textSize.width - 2), (y0 - textSize.height - 2)), colors[b.cl], -1);
        cv::putText(imageORIG, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
        
    }
    return imageORIG;
    // cv::namedWindow("cnet", cv::WINDOW_NORMAL);
    // cv::imshow("cnet", imageOrig);
    // cv::waitKey(10000);
}


void CenternetDetection::update(cv::Mat &imageORIG) {
    
    if(!imageORIG.data) {
        std::cout<<"YOLO: NO IMAGE DATA\n";
        return;
    }  
    TIMER_START
    auto start_t = std::chrono::steady_clock::now();
    auto step_t = std::chrono::steady_clock::now();
    auto end_t = std::chrono::steady_clock::now();
    // -----------------------------------pre-process ------------------------------------------
    // it will resize the images to `224 x 224` in GETTING_STARTED.md
    cv::Size sz = imageORIG.size();
    std::cout<<"image: "<<sz.width<<", "<<sz.height<<std::endl;
    
    float scale = 1.0;
    float new_height = sz.height * scale;
    float new_width = sz.width * scale;
    float c[] = {new_width / 2.0, new_height /2.0};
    float s[2];

    if(sz.width > sz.height){
        s[0] = sz.width * 1.0;
        s[1] = sz.width * 1.0;
    }
    else{
        s[0] = sz.height * 1.0;    
        s[1] = sz.height * 1.0;    
    }

    // ----------- get_affine_transform
    // rot_rad = pi * 0 / 100 --> 0
      
    src.at<float>(0,0)=c[0];
    src.at<float>(0,1)=c[1];
    src.at<float>(1,0)=c[0];
    src.at<float>(1,1)=c[1] + s[0] * -0.5;
    dst.at<float>(0,0)=inp_width * 0.5;
    dst.at<float>(0,1)=inp_height * 0.5;
    dst.at<float>(1,0)=inp_width * 0.5;
    dst.at<float>(1,1)=inp_height * 0.5 +  inp_width * -0.5; 
    
    src.at<float>(2,0)=src.at<float>(1,0) + (-src.at<float>(0,1)+src.at<float>(1,1) );
    src.at<float>(2,1)=src.at<float>(1,1) + (src.at<float>(0,0)-src.at<float>(1,0) );
    dst.at<float>(2,0)=dst.at<float>(1,0) + (-dst.at<float>(0,1)+dst.at<float>(1,1) );
    dst.at<float>(2,1)=dst.at<float>(1,1) + (dst.at<float>(0,0)-dst.at<float>(1,0) );
    // std::cout<<"src: "<<src<<std::endl;
    // std::cout<<"dst: "<<dst<<std::endl;
    cv::Mat trans = cv::getAffineTransform( src, dst );
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME getAffinetr : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    resize(imageORIG, imageF, cv::Size(new_width, new_height));
    sz = imageF.size();
    std::cout<<"size: "<<sz.height<<" "<<sz.width<<" - "<<std::endl;
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME resize: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    cv::warpAffine(imageF, imageF, trans, cv::Size(inp_width, inp_height), cv::INTER_LINEAR );
    
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME warpAffine: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;

    sz = imageF.size();
    std::cout<<"size: "<<sz.height<<" "<<sz.width<<" - "<<std::endl;
    imageF.convertTo(imageF, CV_32FC3, 1/255.0); 
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME convert_to: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    std::cout<<"mean: "<<mean<<", std: "<<stddev<<std::endl;

    dim2 = dim;
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME before split: " << std::chrono::duration_cast<std::chrono::microseconds>(end_t - step_t).count() << " us" << std::endl;
    step_t = end_t;

    //split channels
    cv::split(imageF,bgr);//split source
    
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME split: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    
    for(int i=0; i<3; i++){
        bgr[i] = bgr[i] - mean[i];
        bgr[i] = bgr[i] / stddev[i];
    }

    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME mean std: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;

    //write channels
    for(int i=0; i<dim2.c; i++) {
        int idx = i*imageF.rows*imageF.cols;
        int ch = dim2.c-3 +i;
        std::cout<<"i: "<<i<<", idx: "<<idx<<", ch: "<<ch<<std::endl;
        memcpy((void*)&input[idx], (void*)bgr[ch].data, imageF.rows*imageF.cols*sizeof(dnnType));
    }

    checkCuda(cudaMemcpyAsync(input_d, input, dim2.tot()*sizeof(dnnType), cudaMemcpyHostToDevice));

    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        netRT->infer(dim2, input_d);
        TIMER_STOP
        dim2.print();
    }
    // checkResult(dim2.tot(), input_h, input);
    std::cout<<" --- pre-process ---\n";
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    // ------------------------------------ process --------------------------------------------
    
    rt_out[0] = (dnnType *)netRT->buffersRT[1];
    rt_out[1] = (dnnType *)netRT->buffersRT[2];
    rt_out[2] = (dnnType *)netRT->buffersRT[3]; 
    rt_out[3] = (dnnType *)netRT->buffersRT[4]; 

    activationSIGMOIDForward(rt_out[0], rt_out[0], dim_hm.tot());
    checkCuda( cudaDeviceSynchronize() );
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME sigmoid : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;

    subtractWithThreshold(rt_out[0], rt_out[0] + dim_hm.tot(), rt_out[1], rt_out[0]);

    float *prova;
    checkCuda( cudaMallocHost(&prova, K*sizeof(float)) ); 
    checkCuda( cudaMemcpy(prova, rt_out[0], K*sizeof(float), cudaMemcpyDeviceToHost) );
    std::cout<<"heat:\n";
    for(int i=0; i<K; i++)
        std::cout<<prova[i]<<" ";
    std::cout<<"\n\n\n";
    // for(int i=0; i < dim_hm.tot(); i++){
    //     if(hm_h[i]-hmax_h[i] > toll || hm_h[i]-hmax_h[i] < -toll){
    //         hm_h[i] = 0.0f;
    //     }
    // }
    // checkCuda( cudaFreeHost(hmax_h) );
    std::cout<<" --- hmax ---\n";
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    // ----------- nms end
    // ----------- topk  
      

    // thrust::device_vector<int> ids_d;
    // int ids[dim_hm.h * dim_hm.w];
    // for(int i=0; i<dim_hm.h * dim_hm.w; i++){
    //     ids[i]=i;
    // }
    // std::vector<int> ids2( dim_hm.h * dim_hm.w );
    // for(int i=0; i<dim_hm.h * dim_hm.w; i++){
    //     ids2[i]=i;
    // }
    // int ids2[dim_hm.h * dim_hm.w];
    
    // checkCuda( cudaMemcpy(ids2_d, ids2, dim_hm.h * dim_hm.w*sizeof(int), cudaMemcpyHostToDevice) );
    if(K > dim_hm.h * dim_hm.w){
        printf ("Error topk (K is too large)\n");
        return;
    }

    
    checkCuda( cudaMemcpy(ids_d, ids_, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int), cudaMemcpyHostToDevice) );    
    // checkCuda( cudaMemcpy(ids_2d, ids_2, dim_hm.h * dim_hm.w*sizeof(int), cudaMemcpyHostToDevice) );    
    
    
    // sortAndTopKonDevice(rt_out[0], ids_2d, topk_scores, topk_inds_ , topk_ys_ , topk_xs_ ,dim_hm.h * dim_hm.w, K, dim_hm.c);
    // checkCuda( cudaDeviceSynchronize() );

    // for(int i=0; i<dim_hm.c; i++){
    //     // get the hm->output_dim.h * hm->output_dim.w elements for each channel and sort it. Then find the first 100 elements
    //     // memcpy(ids2, ids, dim_hm.h * dim_hm.w);
    //     sort(rt_out[0]+ i * dim_hm.h * dim_hm.w,
    //             rt_out[0]+ i * dim_hm.h * dim_hm.w + dim_hm.h * dim_hm.w,
    //             ids_d);
    //     // end_t = std::chrono::steady_clock::now();
    //     // std::cout << " TIME sort channel "<<i<<": " << std::chrono::duration_cast<std::chrono::microseconds>(end_t - step_t).count() << " ms" << std::endl;
    //     // step_t = end_t;
    //     topk(rt_out[0]+ i * dim_hm.h * dim_hm.w, ids_d, K, topk_scores + i*K,
    //         topk_inds_ + i*K, topk_ys_ + i*K, topk_xs_ + i*K);
    //     // checkCuda( cudaMemcpy(ids2, ids2_d, dim_hm.h * dim_hm.w*sizeof(int), cudaMemcpyDeviceToHost) );

    //     // for (int j=0; j<dim_hm.h * dim_hm.w; j++) {
    //     //     topk_scores[i*K + count] = hm_h[i * dim_hm.h * dim_hm.w + ids2[j]];
    //     //     topk_inds_[i*K +count] = ids2[j];
    //     //     topk_ys_[i*K +count] = (int)(ids2[j] / width);
    //     //     topk_xs_[i*K +count] = (int)(ids2[j] % width);
    //     //     if(++count == K)
    //     //         break;
    //     // }
    //     // end_t = std::chrono::steady_clock::now();
    //     // std::cout << " TIME topk channel "<<i<<": " << std::chrono::duration_cast<std::chrono::microseconds>(end_t - step_t).count() << " ms" << std::endl;
    //     // step_t = end_t;
    
    // }
    // checkCuda( cudaFree(ids_d ));
    std::cout<<" --- a 100 ---\n";
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME sort topk on 80 channel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    // final

    // sort(topk_scores,
    //     topk_scores +  dim_hm.c * K,
    //     topk_inds_);
    sort(rt_out[0],
            rt_out[0]+dim_hm.tot(),
            ids_d);
    checkCuda( cudaDeviceSynchronize() );   
    int *topk_inds;
    checkCuda( cudaMallocHost(&topk_inds, K*sizeof(int)) ); 
    // checkCuda( cudaMemcpy(topk_inds, ids_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    // for(int i=0; i<K; i++)
    //     std::cout<<topk_inds[i]<<" ";
    // std::cout<<"\n\n\n";

    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME sort channel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;

    // topk(topk_scores, topk_inds_, K, scores_d,
    //     topk_inds_d, topk_ys_d, topk_xs_d);
    topk(rt_out[0], ids_d, K, scores_d,
        topk_inds_d, topk_ys_d, topk_xs_d);
    checkCuda( cudaDeviceSynchronize() );
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME topk channel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    
    
    checkCuda( cudaMemcpy(topk_inds, topk_inds_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    for(int i=0; i<K; i++)
        std::cout<<topk_inds[i]<<" ";
    std::cout<<std::endl;
    
    
    checkCuda( cudaMemcpy(scores, scores_d, K *sizeof(float), cudaMemcpyDeviceToHost) );
    std::cout<<"\n\nscores:\n";
    for(int i=0; i<K;i++)
        std::cout<<scores[i]<<" ";
    std::cout<<std::endl;
    
    
    std::cout<<"\n\n\n";
    topKxyclasses(topk_inds_d, topk_inds_d+K, K, width, dim_hm.w*dim_hm.h, clses_d, inttopk_xs_d, inttopk_ys_d);
    checkCuda( cudaMemcpy(topk_xs_d, (float *)inttopk_xs_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    checkCuda( cudaMemcpy(topk_ys_d, (float *)inttopk_ys_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    
    checkCuda( cudaMemcpy(clses, clses_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    std::cout<<"\ntopk_ids: \n";
    checkCuda( cudaMemcpy(topk_inds, topk_inds_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    for(int i=0; i<K; i++)
        std::cout<<topk_inds[i]<<" ";
    std::cout<<std::endl;
    std::cout<<"\ntopk_clses: \n";
    checkCuda( cudaMemcpy(topk_inds, clses_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    for(int i=0; i<K; i++)
        std::cout<<topk_inds[i]<<" ";
    std::cout<<std::endl;
    std::cout<<"\nxs: \n";
    checkCuda( cudaMemcpy(topk_inds, topk_xs_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    for(int i=0; i<K; i++)
        std::cout<<topk_inds[i]<<" ";
    std::cout<<std::endl;
    std::cout<<"\nys: \n";
    checkCuda( cudaMemcpy(topk_inds, topk_ys_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    for(int i=0; i<K; i++)
        std::cout<<topk_inds[i]<<" ";
    std::cout<<std::endl;
    // return;

    // checkCuda( cudaDeviceSynchronize() );
    // checkCuda( cudaFree(topk_scores) );
    // checkCuda( cudaFree(topk_inds_) );
    // checkCuda( cudaFree(topk_ys_) );
    // checkCuda( cudaFree(topk_xs_) );


    // checkCuda( cudaFree(scores_d) );
    // checkCuda( cudaFree(topk_inds_d) );


    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME clses topk 1 time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    // ----------- topk end 
    
    // dnnType *reg_aus;
    // checkCuda( cudaMallocHost(&reg_aus, dim_reg.tot()*sizeof(dnnType)) ); 
    // checkCuda( cudaMemcpy(reg_aus, rt_out[3], dim_reg.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );    

    // for(int i = 0; i < K; i++){
    //     topk_xs[i] = topk_xs[i] + reg_aus[topk_inds[i]];
    //     topk_ys[i] = topk_ys[i] + reg_aus[topk_inds[i]+dim_reg.h*dim_reg.w];
    // }
    topKxyAddOffset(topk_inds_d, K, dim_reg.h*dim_reg.w, inttopk_xs_d, inttopk_ys_d, topk_xs_d, topk_ys_d, rt_out[3]);
    // checkCuda( cudaDeviceSynchronize() );
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME add offset: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    // checkCuda( cudaFreeHost(reg_aus) );

    // dnnType *wh_aus;
    
    // checkCuda( cudaMemcpy(wh_aus, rt_out[2], dim_wh.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );    
    bboxes(topk_inds_d, K, dim_wh.h*dim_wh.w, topk_xs_d, topk_ys_d, rt_out[2], bbx0_d, bbx1_d, bby0_d, bby1_d);
    // checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaMemcpy(bbx0, bbx0_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bby0, bby0_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bbx1, bbx1_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bby1, bby1_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    // for(int i = 0; i < K; i++){
    //     bboxes[i * 4] = topk_xs[i] - wh_aus[topk_inds[i]] / 2;
    //     bboxes[i * 4 + 1] = topk_ys[i] - wh_aus[topk_inds[i]+dim_reg.h*dim_reg.w] / 2;
    //     bboxes[i * 4 + 2] = topk_xs[i] + wh_aus[topk_inds[i]] / 2;
    //     bboxes[i * 4 + 3] = topk_ys[i] + wh_aus[topk_inds[i]+dim_reg.h*dim_reg.w] / 2;
    // }
    // for(int i = 0; i < K; i++){
    //     std::cout<<"-----\n(x0, y0) = ("<<bbx0<<", "<<bby0<<")\n(x1,y1) = ("<<bbx1<<", "<<bby1<<")\n";
        
    // }

    // checkCuda( cudaFreeHost(wh_aus) );
    // checkCuda( cudaFreeHost(topk_inds) );
    // checkCuda( cudaFreeHost(topk_ys) );
    // checkCuda( cudaFreeHost(topk_xs) );
    std::cout<<" --- bboxes ---\n";
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    // servono [bboxes, scores, clses]
    // checkCuda( cudaDeviceSynchronize() );

    std::cout<<" --- process ---\n";
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    // ---------------------------------- post-process -----------------------------------------
    
    // --------- ctdet_post_process
    // --------- transform_preds 
    src.at<float>(0,0)=c[0];
    src.at<float>(0,1)=c[1];
    src.at<float>(1,0)=c[0];
    src.at<float>(1,1)=c[1] + s[0] * -0.5;
    dst.at<float>(0,0)=width * 0.5;
    dst.at<float>(0,1)=width * 0.5;
    dst.at<float>(1,0)=width * 0.5;
    dst.at<float>(1,1)=width * 0.5 +  width * -0.5; 
    
    src.at<float>(2,0)=src.at<float>(1,0) + (-src.at<float>(0,1)+src.at<float>(1,1) );
    src.at<float>(2,1)=src.at<float>(1,1) + (src.at<float>(0,0)-src.at<float>(1,0) );
    dst.at<float>(2,0)=dst.at<float>(1,0) + (-dst.at<float>(0,1)+dst.at<float>(1,1) );
    dst.at<float>(2,1)=dst.at<float>(1,1) + (dst.at<float>(0,0)-dst.at<float>(1,0) );
    
    
    cv::Mat trans2(cv::Size(3,2), CV_32F);
    trans2 = cv::getAffineTransform( dst, src );
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME getAffineTrans 2: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    cv::Mat new_pt1(cv::Size(1,2), CV_32F);
    cv::Mat new_pt2(cv::Size(1,2), CV_32F);
 

    for(int i = 0; i<K; i++){
        new_pt1.at<float>(0,0)=static_cast<float>(trans2.at<double>(0,0))*bbx0[i] +
                                static_cast<float>(trans2.at<double>(0,1))*bby0[i] +
                                static_cast<float>(trans2.at<double>(0,2))*1.0;
        new_pt1.at<float>(0,1)=static_cast<float>(trans2.at<double>(1,0))*bbx0[i] +
                                static_cast<float>(trans2.at<double>(1,1))*bby0[i] +
                                static_cast<float>(trans2.at<double>(1,2))*1.0;

        new_pt2.at<float>(0,0)=static_cast<float>(trans2.at<double>(0,0))*bbx1[i] +
                                static_cast<float>(trans2.at<double>(0,1))*bby1[i] +
                                static_cast<float>(trans2.at<double>(0,2))*1.0;
        new_pt2.at<float>(0,1)=static_cast<float>(trans2.at<double>(1,0))*bbx1[i] +
                                static_cast<float>(trans2.at<double>(1,1))*bby1[i] +
                                static_cast<float>(trans2.at<double>(1,2))*1.0;

        // std::cout<<"\n new: "<<new_pt1<<" - "<<new_pt2<<std::endl;
        target_coords[i*4] = new_pt1.at<float>(0,0);
        target_coords[i*4+1] = new_pt1.at<float>(0,1);
        target_coords[i*4+2] = new_pt2.at<float>(0,0);
        target_coords[i*4+3] = new_pt2.at<float>(0,1);
        // std::cout<<new_pt1.at<float>(0,0)<<", "<<new_pt1.at<float>(0,1)<<", "<<new_pt2.at<float>(0,0)<<", "<<new_pt2.at<float>(0,1)<<std::endl;
        // std::cout<<"target:cords "<<target_coords[i*4]<<" - "<<target_coords[i*4+1]<<std::endl; 
    }
    
    // int *classes;
    
    detected.clear();
    for(int i = 0; i<classes; i++){
        for(int j=0; j<K; j++)
            if(clses[j] == i){
                if(scores[j] > thresh){
                    std::cout<<"th: "<<scores[j]<<" - cl: "<<clses[j]<<" i: "<<i<<std::endl;
                    //add coco bbox
                    //det[0:4], i, det[4]
                    int x0   = target_coords[j*4];
                    int y0   = target_coords[j*4+1];
                    int x1   = target_coords[j*4+2];
                    int y1   = target_coords[j*4+3];
                    int obj_class = clses[j];
                    float prob = scores[j];
                    // std::cout<<"("<<x0<<", "<<y0<<"),("<<x1<<", "<<y1<<")"<<std::endl;
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
    std::cout<<" --- post_process ---\n";
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
    step_t = end_t;
    std::cout<<"TOTAL: \n";
    TIMER_STOP
}
}}