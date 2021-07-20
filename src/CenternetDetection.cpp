#include "CenternetDetection.h"


namespace tk { namespace dnn {

bool CenternetDetection::init(const std::string& tensor_path, const int n_classes, const int n_batches, const float conf_thresh){
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );
    classes = n_classes;
    nBatches = n_batches;
    confThreshold = conf_thresh;

    dim = netRT->input_dim;

    const char *coco_class_name[] = {
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
    classesNames = std::vector<std::string>(coco_class_name, std::end( coco_class_name));
    
    for(int c=0; c<classes; c++) {
        int offset = c*123457 % classes;
        float r = getColor(2, offset, classes);
        float g = getColor(1, offset, classes);
        float b = getColor(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }
    
    src = cv::Mat(cv::Size(2,3), CV_32F);
    dst = cv::Mat(cv::Size(2,3), CV_32F);
    dst2 = cv::Mat(cv::Size(2,3), CV_32F);
    trans = cv::Mat(cv::Size(3,2), CV_32F);
    trans2 = cv::Mat(cv::Size(3,2), CV_32F);

    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*netRT->input_dim.tot() * nBatches));

    dim_hm = tk::dnn::dataDim_t(1, 80, 128, 128, 1);
    dim_wh = tk::dnn::dataDim_t(1, 2, 128, 128, 1);
    dim_reg = tk::dnn::dataDim_t(1, 2, 128, 128, 1);

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

    checkCuda( cudaMalloc(&topk_inds_d, K *sizeof(int)) );
    checkCuda( cudaMalloc(&topk_ys_d, K *sizeof(float)) );     
    checkCuda( cudaMalloc(&topk_xs_d, K *sizeof(float)) ); 
    checkCuda( cudaMalloc(&inttopk_ys_d, K *sizeof(int)) );
    checkCuda( cudaMalloc(&inttopk_xs_d, K *sizeof(int)) );

    checkCuda( cudaMallocHost(&bbx0, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bby0, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bbx1, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bby1, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bbx0_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bby0_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bbx1_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bby1_d, K * sizeof(float)) ); 

    checkCuda( cudaMallocHost(&target_coords, 4 * K *sizeof(float)) );

#ifdef OPENCV_CUDACONTRIB

    checkCuda( cudaMalloc(&mean_d, 3 * sizeof(float)) );
    checkCuda( cudaMalloc(&stddev_d, 3 * sizeof(float)) );
    float mean[3] = {0.408, 0.447, 0.47};
    float stddev[3] = {0.289, 0.274, 0.278};
    
    checkCuda(cudaMemcpy(mean_d, mean, 3*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(stddev_d, stddev, 3*sizeof(float), cudaMemcpyHostToDevice));
#else
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*netRT->input_dim.tot()* nBatches));
    mean << 0.408, 0.447, 0.47;
    stddev << 0.289, 0.274, 0.278;
#endif

    checkCuda( cudaMalloc(&d_ptrs, dim.c * dim.h*dim.w * sizeof(float)) );

    // Alloc array used in the kernel 
    checkCuda( cudaMalloc(&src_out, K *sizeof(float)) );
    checkCuda( cudaMalloc(&ids_out, K *sizeof(int)) );

    dst2.at<float>(0,0)=width * 0.5;
    dst2.at<float>(0,1)=width * 0.5;
    dst2.at<float>(1,0)=width * 0.5;
    dst2.at<float>(1,1)=width * 0.5 +  width * -0.5; 
    
    dst2.at<float>(2,0)=dst2.at<float>(1,0) + (-dst2.at<float>(0,1)+dst2.at<float>(1,1) );
    dst2.at<float>(2,1)=dst2.at<float>(1,1) + (dst2.at<float>(0,0)-dst2.at<float>(1,0) );
    
}


void CenternetDetection::preprocess(cv::Mat &frame, const int bi){
     // -----------------------------------pre-process ------------------------------------------
    
    // auto start_t = std::chrono::steady_clock::now();
    // auto step_t = std::chrono::steady_clock::now();
    // auto end_t = std::chrono::steady_clock::now();
    cv::Size sz = originalSize[bi];
    // std::cout<<"image: "<<sz.width<<", "<<sz.height<<std::endl;
    cv::Size sz_old;
    float scale = 1.0;
    float new_height = sz.height * scale;
    float new_width = sz.width * scale;
    if(sz.height != sz_old.height && sz.width != sz_old.width){
        float c[] = {new_width / 2.0f, new_height /2.0f};
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
        dst.at<float>(0,0)=netRT->input_dim.w * 0.5;
        dst.at<float>(0,1)=netRT->input_dim.h * 0.5;
        dst.at<float>(1,0)=netRT->input_dim.w * 0.5;
        dst.at<float>(1,1)=netRT->input_dim.h * 0.5 +  netRT->input_dim.w * -0.5; 
        
        src.at<float>(2,0)=src.at<float>(1,0) + (-src.at<float>(0,1)+src.at<float>(1,1) );
        src.at<float>(2,1)=src.at<float>(1,1) + (src.at<float>(0,0)-src.at<float>(1,0) );
        dst.at<float>(2,0)=dst.at<float>(1,0) + (-dst.at<float>(0,1)+dst.at<float>(1,1) );
        dst.at<float>(2,1)=dst.at<float>(1,1) + (dst.at<float>(0,0)-dst.at<float>(1,0) );

        trans = cv::getAffineTransform( src, dst );
        // end_t = std::chrono::steady_clock::now();
        // std::cout << " TIME gett affine trans: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
        // step_t = end_t;

        trans2 = cv::getAffineTransform( dst2, src );
        // end_t = std::chrono::steady_clock::now();
        // std::cout << " TIME getAffineTrans 2: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
        // step_t = end_t;
    }
    sz_old = sz;
#ifdef OPENCV_CUDACONTRIB
    cv::cuda::GpuMat im_Orig; 
    cv::cuda::GpuMat imageF1_d, imageF2_d;
        
    im_Orig = cv::cuda::GpuMat(frame);
    cv::cuda::resize (im_Orig, imageF1_d, cv::Size(new_width, new_height)); 
    checkCuda( cudaDeviceSynchronize() );
    
    sz = imageF1_d.size();
    // std::cout<<"size: "<<sz.height<<" "<<sz.width<<" - "<<std::endl;
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME resize: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;
    
    cv::cuda::warpAffine(imageF1_d, imageF2_d, trans, cv::Size(netRT->input_dim.w, netRT->input_dim.h), cv::INTER_LINEAR );
    checkCuda( cudaDeviceSynchronize() );
    
    imageF2_d.convertTo(imageF1_d, CV_32FC3, 1/255.0); 
    checkCuda( cudaDeviceSynchronize() );
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME convert: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;
    
    dim2 = dim;
    cv::cuda::GpuMat bgr[3]; 
    cv::cuda::split(imageF1_d,bgr);//split source
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME split: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;

    for(int i=0; i<dim.c; i++)
        checkCuda( cudaMemcpy(d_ptrs + i*dim.h * dim.w, (float*)bgr[i].data, dim.h * dim.w * sizeof(float), cudaMemcpyDeviceToDevice) );
        
    normalize(d_ptrs, dim.c, dim.h, dim.w, mean_d, stddev_d);
    
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME normalize: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;

    checkCuda(cudaMemcpy(input_d+ netRT->input_dim.tot()*bi, d_ptrs, dim2.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));
    
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME Memcpy to input_d: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;
#else
 
    cv::Mat imageF;
    resize(frame, imageF, cv::Size(new_width, new_height));
    sz = imageF.size();
    // std::cout<<"size: "<<sz.height<<" "<<sz.width<<" - "<<std::endl;
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME resize: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;

    cv::Mat trans = cv::getAffineTransform( src, dst );
    cv::warpAffine(imageF, imageF, trans, cv::Size(netRT->input_dim.w, netRT->input_dim.h), cv::INTER_LINEAR );
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME warpAffine: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;

    sz = imageF.size();
    // std::cout<<"size: "<<sz.height<<" "<<sz.width<<" - "<<std::endl;
    imageF.convertTo(imageF, CV_32FC3, 1/255.0); 
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME convertto: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;

    dim2 = dim;
    //split channels
    cv::Mat bgr[3]; 
    cv::split(imageF,bgr);//split source
    for(int i=0; i<3; i++){
        bgr[i] = bgr[i] - mean[i];
        bgr[i] = bgr[i] / stddev[i];
    }

    //write channels
    for(int i=0; i<dim2.c; i++) {
        int idx = i*imageF.rows*imageF.cols;
        int ch = dim2.c-3 +i;
        // std::cout<<"i: "<<i<<", idx: "<<idx<<", ch: "<<ch<<std::endl;
        memcpy((void*)&input[idx+ netRT->input_dim.tot()*bi], (void*)bgr[ch].data, imageF.rows*imageF.cols*sizeof(dnnType));
    }
    checkCuda(cudaMemcpyAsync(input_d+ netRT->input_dim.tot()*bi, input+ netRT->input_dim.tot()*bi, dim2.tot()*sizeof(dnnType), cudaMemcpyHostToDevice));
#endif
}

void CenternetDetection::postprocess(const int bi, const bool mAP){
    dnnType *rt_out[4];
    rt_out[0] = (dnnType *)netRT->buffersRT[1]+ netRT->buffersDIM[1].tot()*bi;
    rt_out[1] = (dnnType *)netRT->buffersRT[2]+ netRT->buffersDIM[2].tot()*bi;
    rt_out[2] = (dnnType *)netRT->buffersRT[3]+ netRT->buffersDIM[3].tot()*bi; 
    rt_out[3] = (dnnType *)netRT->buffersRT[4]+ netRT->buffersDIM[4].tot()*bi; 

    // auto start_t = std::chrono::steady_clock::now();
    // auto step_t = std::chrono::steady_clock::now();
    // auto end_t = std::chrono::steady_clock::now();
    // ------------------------------------ process --------------------------------------------
    activationSIGMOIDForward(rt_out[0], rt_out[0], dim_hm.tot());
    checkCuda( cudaDeviceSynchronize() );

    subtractWithThreshold(rt_out[0], rt_out[0] + dim_hm.tot(), rt_out[1], rt_out[0], op);
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME threshold: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;
    // ----------- nms end
    // ----------- topk  
      
    if(K > dim_hm.h * dim_hm.w){
        printf ("Error topk (K is too large)\n");
        return;
    }
    
    checkCuda( cudaMemcpy(ids_d, ids_, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int), cudaMemcpyHostToDevice) );    

    sort(rt_out[0],rt_out[0]+dim_hm.tot(),ids_d);
    checkCuda( cudaDeviceSynchronize() );   
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME sort: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;

    topk(rt_out[0], ids_d, K, scores_d, topk_inds_d, topk_ys_d, topk_xs_d);
    checkCuda( cudaDeviceSynchronize() );    

    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME topk: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;
        
    checkCuda( cudaMemcpy(scores, scores_d, K *sizeof(float), cudaMemcpyDeviceToHost) );

    topKxyclasses(topk_inds_d, topk_inds_d+K, K, width, dim_hm.w*dim_hm.h, clses_d, inttopk_xs_d, inttopk_ys_d);
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME topk x y clses 2: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;

    checkCuda( cudaMemcpy(topk_xs_d, (float *)inttopk_xs_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    checkCuda( cudaMemcpy(topk_ys_d, (float *)inttopk_ys_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    
    checkCuda( cudaMemcpy(clses, clses_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    
    // ----------- topk end 
    
    topKxyAddOffset(topk_inds_d, K, dim_reg.h*dim_reg.w, inttopk_xs_d, inttopk_ys_d, topk_xs_d, topk_ys_d, rt_out[3], src_out, ids_out);
    // checkCuda( cudaDeviceSynchronize() );
    
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME add offset: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;
    
    bboxes(topk_inds_d, K, dim_wh.h*dim_wh.w, topk_xs_d, topk_ys_d, rt_out[2], bbx0_d, bbx1_d, bby0_d, bby1_d, src_out, ids_out);
    // checkCuda( cudaDeviceSynchronize() );
    
    checkCuda( cudaMemcpy(bbx0, bbx0_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bby0, bby0_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bbx1, bbx1_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bby1, bby1_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME bboxes: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;

    // ---------------------------------- post-process -----------------------------------------
    
    // --------- ctdet_post_process
    // --------- transform_preds 
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

        target_coords[i*4] = new_pt1.at<float>(0,0);
        target_coords[i*4+1] = new_pt1.at<float>(0,1);
        target_coords[i*4+2] = new_pt2.at<float>(0,0);
        target_coords[i*4+3] = new_pt2.at<float>(0,1);
    }
       
    detected.clear();
    for(int i = 0; i<classes; i++){
        for(int j=0; j<K; j++)
            if(clses[j] == i){
                if(scores[j] > confThreshold){
                    // std::cout<<"th: "<<scores[j]<<" - cl: "<<clses[j]<<" i: "<<i<<std::endl;
                    //add coco bbox
                    //det[0:4], i, det[4]
                    float x0   = target_coords[j*4];
                    float y0   = target_coords[j*4+1];
                    float x1   = target_coords[j*4+2];
                    float y1   = target_coords[j*4+3];
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
    
    batchDetected.push_back(detected);
    // end_t = std::chrono::steady_clock::now();
    // std::cout << " TIME detections: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    // step_t = end_t;
}


}}


