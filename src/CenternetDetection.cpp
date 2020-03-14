#ifndef CENTERNETDETECTION_H
#define CENTERNETDETECTION_H

#include "CenternetDetection.h"
#include "opencv2/imgproc/imgproc.hpp"
// #include <opencv2/cudawarping.hpp>
// #include <opencv2/cudaarithm.hpp>



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
    
    dim = tk::dnn::dataDim_t(1, 3, 512, 512, 1);
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
    dst2 = cv::Mat(cv::Size(2,3), CV_32F);
    trans = cv::Mat(cv::Size(3,2), CV_32F);
    trans2 = cv::Mat(cv::Size(3,2), CV_32F);
    // dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);

    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*netRT->input_dim.tot()));

    // dim_hm = tk::dnn::dataDim_t(1, 80, 56, 56, 1);
    // dim_wh = tk::dnn::dataDim_t(1, 2, 56, 56, 1);
    // dim_reg = tk::dnn::dataDim_t(1, 2, 56, 56, 1);
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

    checkCuda( cudaMalloc(&mean_d, 3 * sizeof(float)) );
    checkCuda( cudaMalloc(&stddev_d, 3 * sizeof(float)) );
    float mean[3] = {0.408, 0.447, 0.47};
    float stddev[3] = {0.289, 0.274, 0.278};
    
    checkCuda(cudaMemcpy(mean_d, mean, 3*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(stddev_d, stddev, 3*sizeof(float), cudaMemcpyHostToDevice));

    checkCuda( cudaMalloc(&d_ptrs, dim.c * dim.h*dim.w * sizeof(float)) );
    // mean << 0.408, 0.447, 0.47;
    // stddev << 0.289, 0.274, 0.278;

    // Alloc array used in the kernel 
    checkCuda( cudaMalloc(&src_out, K *sizeof(float)) );
    checkCuda( cudaMalloc(&ids_out, K *sizeof(int)) );
    // checkCuda( cudaFree(src_out) );
    // checkCuda( cudaFree(ids_out) );
    dst2.at<float>(0,0)=width * 0.5;
    dst2.at<float>(0,1)=width * 0.5;
    dst2.at<float>(1,0)=width * 0.5;
    dst2.at<float>(1,1)=width * 0.5 +  width * -0.5; 
    
    dst2.at<float>(2,0)=dst2.at<float>(1,0) + (-dst2.at<float>(0,1)+dst2.at<float>(1,1) );
    dst2.at<float>(2,1)=dst2.at<float>(1,1) + (dst2.at<float>(0,0)-dst2.at<float>(1,0) );
    
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
        std::cout<<"CENTERNET: NO IMAGE DATA\n";
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
    if(sz.height != sz_old.height && sz.width != sz_old.width){
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

        trans = cv::getAffineTransform( src, dst );
        end_t = std::chrono::steady_clock::now();
        std::cout << " TIME gett affine trans: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
        step_t = end_t;

        trans2 = cv::getAffineTransform( dst2, src );

        end_t = std::chrono::steady_clock::now();
        std::cout << " TIME getAffineTrans 2: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
        step_t = end_t;
    }
    sz_old = sz;
    cv::cuda::GpuMat im_Orig; 
    im_Orig = cv::cuda::GpuMat(imageORIG);
    // cv::cuda::resize (im_Orig, imageF1_d, cv::Size(new_width, new_height)); 
    checkCuda( cudaDeviceSynchronize() );
    
    sz = imageF1_d.size();
    std::cout<<"size: "<<sz.height<<" "<<sz.width<<" - "<<std::endl;
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME resize: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;
    
    // cv::cuda::warpAffine(imageF1_d, imageF2_d, trans, cv::Size(inp_width, inp_height), cv::INTER_LINEAR );
    checkCuda( cudaDeviceSynchronize() );
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME warpAffine: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;
    
    imageF2_d.convertTo(imageF1_d, CV_32FC3, 1/255.0); 
    checkCuda( cudaDeviceSynchronize() );
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME convert: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;
    
    dim2 = dim;
    // cv::cuda::split(imageF1_d,bgr);//split source
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME split: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;

    for(int i=0; i<dim.c; i++)
        checkCuda( cudaMemcpy(d_ptrs + i*dim.h * dim.w, (float*)bgr[i].data, dim.h * dim.w * sizeof(float), cudaMemcpyDeviceToDevice) );
        
    normalize(d_ptrs, dim.c, dim.h, dim.w, mean_d, stddev_d);
    
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME normalize: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;

    checkCuda(cudaMemcpy(input_d, d_ptrs, dim2.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));
    
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME Memcpy to input_d: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;

    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        netRT->infer(dim2, input_d);
        TIMER_STOP
        dim2.print();
    }
    step_t = std::chrono::steady_clock::now();
    
    // ------------------------------------ process --------------------------------------------
    
    rt_out[0] = (dnnType *)netRT->buffersRT[1];
    rt_out[1] = (dnnType *)netRT->buffersRT[2];
    rt_out[2] = (dnnType *)netRT->buffersRT[3]; 
    rt_out[3] = (dnnType *)netRT->buffersRT[4]; 
    activationSIGMOIDForward(rt_out[0], rt_out[0], dim_hm.tot());
    checkCuda( cudaDeviceSynchronize() );

    subtractWithThreshold(rt_out[0], rt_out[0] + dim_hm.tot(), rt_out[1], rt_out[0], op);

    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME threshold: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;
    // ----------- nms end
    // ----------- topk  
      
    if(K > dim_hm.h * dim_hm.w){
        printf ("Error topk (K is too large)\n");
        return;
    }
    
    checkCuda( cudaMemcpy(ids_d, ids_, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int), cudaMemcpyHostToDevice) );    

    sort(rt_out[0],
            rt_out[0]+dim_hm.tot(),
            ids_d);
    checkCuda( cudaDeviceSynchronize() );   
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME sort: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;

    topk(rt_out[0], ids_d, K, scores_d,
        topk_inds_d, topk_ys_d, topk_xs_d);
    checkCuda( cudaDeviceSynchronize() );    

    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME topk: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;
        
    checkCuda( cudaMemcpy(scores, scores_d, K *sizeof(float), cudaMemcpyDeviceToHost) );

    topKxyclasses(topk_inds_d, topk_inds_d+K, K, width, dim_hm.w*dim_hm.h, clses_d, inttopk_xs_d, inttopk_ys_d);
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME topk x y clses 2: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;

    checkCuda( cudaMemcpy(topk_xs_d, (float *)inttopk_xs_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    checkCuda( cudaMemcpy(topk_ys_d, (float *)inttopk_ys_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    
    checkCuda( cudaMemcpy(clses, clses_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    
    // ----------- topk end 
    
    topKxyAddOffset(topk_inds_d, K, dim_reg.h*dim_reg.w, inttopk_xs_d, inttopk_ys_d, topk_xs_d, topk_ys_d, rt_out[3], src_out, ids_out);
    // checkCuda( cudaDeviceSynchronize() );
    
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME add offset: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;
    
    bboxes(topk_inds_d, K, dim_wh.h*dim_wh.w, topk_xs_d, topk_ys_d, rt_out[2], bbx0_d, bbx1_d, bby0_d, bby1_d, src_out, ids_out);
    // checkCuda( cudaDeviceSynchronize() );
    
    checkCuda( cudaMemcpy(bbx0, bbx0_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bby0, bby0_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bbx1, bbx1_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bby1, bby1_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME bboxes: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;

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
                if(scores[j] > thresh){
                    // std::cout<<"th: "<<scores[j]<<" - cl: "<<clses[j]<<" i: "<<i<<std::endl;
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
    
    end_t = std::chrono::steady_clock::now();
    std::cout << " TIME detections: " << std::chrono::duration_cast<std::chrono:: microseconds>(end_t - step_t).count() << "  us" << std::endl;
    step_t = end_t;

    std::cout<<"TOTAL: \n";
    TIMER_STOP
    stats.push_back(t_ns);
}
}}

#endif /*CENTERNETDETECTION_H*/
