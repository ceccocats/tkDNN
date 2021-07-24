#include "CenternetDetection3D.h"


namespace tk { namespace dnn {

bool CenternetDetection3D::init(const std::string& tensor_path, const int n_classes, const int n_batches, 
                                const float conf_thresh, const std::vector<cv::Mat>& k_calibs) {
    std::cout<<(tensor_path).c_str()<<"\n";
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );
    classes = n_classes;
    nBatches = n_batches;
    confThreshold = conf_thresh;
    inputCalibs = k_calibs;
    dim = netRT->input_dim;

    const char *kitti_class_name[] = {
            "person", "car", "bicycle"};
    classesNames = std::vector<std::string>(kitti_class_name, std::end( kitti_class_name));
    
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

    dim_hm = tk::dnn::dataDim_t(1, 3, 128, 128, 1);
    dim_wh = tk::dnn::dataDim_t(1, 2, 128, 128, 1);
    dim_reg = tk::dnn::dataDim_t(1, 2, 128, 128, 1);
    dim_dep = tk::dnn::dataDim_t(1, 1, 128, 128, 1);
    dim_rot = tk::dnn::dataDim_t(1, 8, 128, 128, 1);
    dim_dim = tk::dnn::dataDim_t(1, 3, 128, 128, 1);

    checkCuda( cudaMalloc(&topk_scores, dim_hm.c * K *sizeof(float)) );
    checkCuda( cudaMalloc(&topk_inds_, dim_hm.c * K *sizeof(int)) );      
    checkCuda( cudaMalloc(&topk_ys_, dim_hm.c * K *sizeof(float)) );      
    checkCuda( cudaMalloc(&topk_xs_, dim_hm.c * K *sizeof(float)) );    
    checkCuda( cudaMalloc(&ids_d, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int)) );
    checkCuda( cudaMallocHost(&ids_, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int)) );
    for(int i =0; i<dim_hm.c * dim_hm.h * dim_hm.w; i++){
        ids_[i] = i;
    }
    
    checkCuda( cudaMalloc(&ones, dim_dep.c * dim_dep.h * dim_dep.w * sizeof(float)) );
    float *ones_h;
    checkCuda( cudaMallocHost(&ones_h, dim_dep.c * dim_dep.h * dim_dep.w * sizeof(float)) );
    for(int i=0; i<dim_dep.c * dim_dep.h * dim_dep.w; i++)
        ones_h[i]=1.0f;
    checkCuda( cudaMemcpy(ones, ones_h, dim_dep.c * dim_dep.h * dim_dep.w * sizeof(float), cudaMemcpyHostToDevice) );
    checkCuda( cudaFreeHost(ones_h) );

    checkCuda( cudaMallocHost(&scores, K *sizeof(float)) );
    checkCuda( cudaMalloc(&scores_d, K *sizeof(float)) );

    checkCuda( cudaMallocHost(&clses, K *sizeof(int)) );
    checkCuda( cudaMalloc(&clses_d, K *sizeof(int)) );

    checkCuda( cudaMalloc(&topk_inds_d, K *sizeof(int)) );
    checkCuda( cudaMalloc(&topk_ys_d, K *sizeof(float)) );     
    checkCuda( cudaMalloc(&topk_xs_d, K *sizeof(float)) ); 
    checkCuda( cudaMalloc(&inttopk_ys_d, K *sizeof(int)) );
    checkCuda( cudaMalloc(&inttopk_xs_d, K *sizeof(int)) );

    checkCuda( cudaMallocHost(&xs, K  * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&ys, K  * sizeof(float)) ); 

    checkCuda( cudaMallocHost(&dep, K * dim_dep.c * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&rot, K * dim_rot.c * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&dim_, K * dim_dim.c * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&wh, K * dim_wh.c * sizeof(float)) ); 
    checkCuda( cudaMalloc(&dep_d, K * dim_dep.c * sizeof(float)) ); 
    checkCuda( cudaMalloc(&rot_d, K * dim_rot.c * sizeof(float)) ); 
    checkCuda( cudaMalloc(&dim_d, K * dim_dim.c * sizeof(float)) ); 
    checkCuda( cudaMalloc(&wh_d, K * dim_wh.c * sizeof(float)) ); 

    checkCuda( cudaMallocHost(&target_coords, 4 * K *sizeof(float)) );

#ifdef OPENCV_CUDACONTRIB

    checkCuda( cudaMalloc(&mean_d, 3 * sizeof(float)) );
    checkCuda( cudaMalloc(&stddev_d, 3 * sizeof(float)) );
    float mean[3] = {0.485, 0.456, 0.406};
    float stddev[3] = {0.229, 0.224, 0.225};
    
    checkCuda(cudaMemcpy(mean_d, mean, 3*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(stddev_d, stddev, 3*sizeof(float), cudaMemcpyHostToDevice));
#else
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*netRT->input_dim.tot() * nBatches));
    mean << 0.485, 0.456, 0.406;
    stddev << 0.229, 0.224, 0.225;
#endif

    for(int bi=0; bi<nBatches; bi++) {
        cv::Mat calibs_ = cv::Mat::zeros(cv::Size(4,3), CV_32F);        
        if(inputCalibs.size() == 0 || inputCalibs[bi].empty()) {
            calibs_.at<float>(0,0) = 707.0493;
            calibs_.at<float>(0,2) = 604.0814;
            calibs_.at<float>(1,1) = 707.0493;
            calibs_.at<float>(1,2) = 180.5066;
            calibs_.at<float>(0,3) = 45.75831;
            calibs_.at<float>(1,3) = -0.3454157;
            calibs_.at<float>(2,2) = 1.0;
            calibs_.at<float>(2,3) = 0.004981016;
        }
        else {
            calibs_.at<float>(0,0) = inputCalibs[bi].at<float>(0,0);// * (1440.0/dim.w);// / 1440;
            calibs_.at<float>(0,2) = inputCalibs[bi].at<float>(0,2);// * (1440.0/dim.w);// / 1440;
            calibs_.at<float>(1,1) = inputCalibs[bi].at<float>(1,1);// * (1080.0/dim.h);//dim.h / 1080;
            calibs_.at<float>(1,2) = inputCalibs[bi].at<float>(1,2);// * (1080.0/dim.h);//dim.h / 1080;
            calibs_.at<float>(2,2) = 1.0;
        }
        // calibs_.at<float>(0,3) = 45.75831;
        // calibs_.at<float>(1,3) = -0.3454157;
        // calibs_.at<float>(2,2) = 1.0;
        // calibs_.at<float>(2,3) = 0.004981016;
        calibs.push_back(calibs_);
    }

    r = cv::Mat(cv::Size(3,3), CV_32F);
    r.at<float>(0,1) = 0.0;
    r.at<float>(1,0) = 0.0;
    r.at<float>(1,1) = 1.0;
    r.at<float>(1,2) = 0.0;
    r.at<float>(2,1) = 0.0;
                        
    corners = cv::Mat(cv::Size(8,3), CV_32F);
    corners.at<float>(1,0) = 0.0;
    corners.at<float>(1,1) = 0.0;
    corners.at<float>(1,2) = 0.0;
    corners.at<float>(1,3) = 0.0;

    pts3DHomo = cv::Mat(cv::Size(8,4), CV_32F);
    pts3DHomo.at<float>(3,0) = 1.0;
    pts3DHomo.at<float>(3,1) = 1.0;
    pts3DHomo.at<float>(3,2) = 1.0;
    pts3DHomo.at<float>(3,3) = 1.0;
    pts3DHomo.at<float>(3,4) = 1.0;
    pts3DHomo.at<float>(3,5) = 1.0;
    pts3DHomo.at<float>(3,6) = 1.0;
    pts3DHomo.at<float>(3,7) = 1.0;
    
    checkCuda( cudaMalloc(&d_ptrs, dim.c * dim.h*dim.w * sizeof(float)) );

    // Alloc array used in the kernel 
    checkCuda( cudaMalloc(&srcOut, K *sizeof(float)) );
    checkCuda( cudaMalloc(&idsOut, K *sizeof(int)) );

    dst2.at<float>(0,0)=width * 0.5;
    dst2.at<float>(0,1)=width * 0.5;
    dst2.at<float>(1,0)=width * 0.5;
    dst2.at<float>(1,1)=width * 0.5 +  width * -0.5; 
    
    dst2.at<float>(2,0)=dst2.at<float>(1,0) + (-dst2.at<float>(0,1)+dst2.at<float>(1,1) );
    dst2.at<float>(2,1)=dst2.at<float>(1,1) + (dst2.at<float>(0,0)-dst2.at<float>(1,0) );

    faceId.push_back({0,1,5,4});
    faceId.push_back({1,2,6, 5});
    faceId.push_back({2,3,7,6});
    faceId.push_back({3,0,4,7});
    // ([[0,1,5,4], [1,2,6, 5], [2,3,7,6], [3,0,4,7]]);
}

void CenternetDetection3D::preprocess(cv::Mat &frame, const int bi){    
    cv::Size sz = originalSize[bi];
    float new_height = dim.h;//sz.height * scale;
    float new_width = dim.w;//sz.width * scale;
    if(sz.height != sz_old.height && sz.width != sz_old.width){

        if(inputCalibs.size() == 0 || inputCalibs[bi].empty()) {
            calibs[bi].at<float>(0,2) = new_width / 2.0f;
            calibs[bi].at<float>(1,2) = new_height /2.0f;
        }
        else {
            calibs[bi].at<float>(0,0) = inputCalibs[bi].at<float>(0,0) * 2.0 * dim.w / sz.width;
            calibs[bi].at<float>(0,2) = inputCalibs[bi].at<float>(0,2) * dim.w / sz.width ;
            calibs[bi].at<float>(1,1) = inputCalibs[bi].at<float>(1,1) * 2.0 * dim.h / sz.height;
            calibs[bi].at<float>(1,2) = inputCalibs[bi].at<float>(1,2) * dim.h / sz.height;
        }
        float c[] = {new_width / 2.0f, new_height /2.0f};
        float s[] = {new_width, new_height};
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
    // std::cout<<"OPENCV CPMTROB\n";
    cv::cuda::GpuMat im_Orig; 
    cv::cuda::GpuMat imageF1_d, imageF2_d;
        
    im_Orig = cv::cuda::GpuMat(frame);
    cv::cuda::resize (im_Orig, imageF1_d, cv::Size(dim.w, dim.h));//cv::Size(new_width, new_height)); 
    // imageF1_d = im_Orig;
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
    // std::cout<<"NO OPENCV CPMTROB\n";
    cv::Mat imageF;
    resize(frame, imageF, cv::Size(dim.w, dim.h));//cv::Size(new_width, new_height));
    // imageF = frame;
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

void CenternetDetection3D::postprocess(const int bi, const bool mAP) {
    dnnType *rt_out[7];
    rt_out[0] = (dnnType *)netRT->buffersRT[1]+ netRT->buffersDIM[1].tot()*bi;
    rt_out[1] = (dnnType *)netRT->buffersRT[2]+ netRT->buffersDIM[2].tot()*bi;
    rt_out[2] = (dnnType *)netRT->buffersRT[3]+ netRT->buffersDIM[3].tot()*bi; 
    rt_out[3] = (dnnType *)netRT->buffersRT[4]+ netRT->buffersDIM[4].tot()*bi; 
    rt_out[4] = (dnnType *)netRT->buffersRT[5]+ netRT->buffersDIM[5].tot()*bi; 
    rt_out[5] = (dnnType *)netRT->buffersRT[6]+ netRT->buffersDIM[6].tot()*bi; 
    rt_out[6] = (dnnType *)netRT->buffersRT[7]+ netRT->buffersDIM[7].tot()*bi; 
    
    // ------------------------------------ process --------------------------------------------
    activationSIGMOIDForward(rt_out[0], rt_out[0], dim_hm.tot());
    checkCuda( cudaDeviceSynchronize() );

    // output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    activationSIGMOIDForward(rt_out[4], rt_out[4], dim_dep.tot());
    checkCuda( cudaDeviceSynchronize() );
    transformDep(ones, ones + dim_dep.tot(), rt_out[4], rt_out[4] + dim_dep.tot());
    checkCuda( cudaDeviceSynchronize() );

    subtractWithThreshold(rt_out[0], rt_out[0] + dim_hm.tot(), rt_out[1], rt_out[0], op);

    // ----------- nms end
    // ----------- topk  
      
    if(K > dim_hm.h * dim_hm.w){
        printf ("Error topk (K is too large)\n");
        return;
    }
    
    checkCuda( cudaMemcpy(ids_d, ids_, dim_hm.c * dim_hm.h * dim_hm.w*sizeof(int), cudaMemcpyHostToDevice) );    

    sort(rt_out[0],rt_out[0]+dim_hm.tot(),ids_d);
    checkCuda( cudaDeviceSynchronize() );   

    topk(rt_out[0], ids_d, K, scores_d, topk_inds_d, topk_ys_d, topk_xs_d);
    checkCuda( cudaDeviceSynchronize() );    

    checkCuda( cudaMemcpy(scores, scores_d, K *sizeof(float), cudaMemcpyDeviceToHost) );

    topKxyclasses(topk_inds_d, topk_inds_d+K, K, width, dim_hm.w*dim_hm.h, clses_d, inttopk_xs_d, inttopk_ys_d);

    checkCuda( cudaMemcpy(topk_xs_d, (float *)inttopk_xs_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    checkCuda( cudaMemcpy(topk_ys_d, (float *)inttopk_ys_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    
    checkCuda( cudaMemcpy(clses, clses_d, K*sizeof(int), cudaMemcpyDeviceToHost) );

    // ----------- topk end 
    
    topKxyAddOffset(topk_inds_d, K, dim_reg.h*dim_reg.w, inttopk_xs_d, inttopk_ys_d, topk_xs_d, topk_ys_d, rt_out[3], srcOut, idsOut);
    // checkCuda( cudaDeviceSynchronize() );
    
   
    getRecordsFromTopKId(topk_inds_d, K, dim_dep.c, dim_dep.h * dim_dep.w, rt_out[4], dep_d, idsOut);
    checkCuda( cudaMemcpy(dep, dep_d, K * dim_dep.c * sizeof(float), cudaMemcpyDeviceToHost) );
    
    getRecordsFromTopKId(topk_inds_d, K, dim_rot.c, dim_rot.h * dim_rot.w, rt_out[5], rot_d, idsOut);
    checkCuda( cudaMemcpy(rot, rot_d, K * dim_rot.c * sizeof(float), cudaMemcpyDeviceToHost) );
    
    getRecordsFromTopKId(topk_inds_d, K, dim_dim.c, dim_dim.h * dim_dim.w, rt_out[6], dim_d, idsOut);
    checkCuda( cudaMemcpy(dim_, dim_d, K * dim_dim.c * sizeof(float), cudaMemcpyDeviceToHost) );

    getRecordsFromTopKId(topk_inds_d, K, dim_wh.c, dim_wh.h * dim_wh.w, rt_out[2], wh_d, idsOut);
    checkCuda( cudaMemcpy(wh, wh_d, K * dim_wh.c * sizeof(float), cudaMemcpyDeviceToHost) );
   
    checkCuda( cudaMemcpy(xs, topk_xs_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(ys, topk_ys_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
  
    // ---------------------------------- post-process -----------------------------------------
    
    // ddd_post_process_2d
    cv::Mat new_pt1(cv::Size(1,2), CV_32F);
    cv::Mat new_pt2(cv::Size(1,2), CV_32F); 

    for(int i = 0; i<K; i++){
        new_pt1.at<float>(0,0)=static_cast<float>(trans2.at<double>(0,0))*xs[i] +
                                static_cast<float>(trans2.at<double>(0,1))*ys[i] +
                                static_cast<float>(trans2.at<double>(0,2))*1.0;
        new_pt1.at<float>(0,1)=static_cast<float>(trans2.at<double>(1,0))*xs[i] +
                                static_cast<float>(trans2.at<double>(1,1))*ys[i] +
                                static_cast<float>(trans2.at<double>(1,2))*1.0;

        new_pt2.at<float>(0,0)=static_cast<float>(trans2.at<double>(0,0))*wh[i] +
                                static_cast<float>(trans2.at<double>(0,1))*wh[K+i] +
                                static_cast<float>(trans2.at<double>(0,2))*1.0;
        new_pt2.at<float>(0,1)=static_cast<float>(trans2.at<double>(1,0))*wh[i] +
                                static_cast<float>(trans2.at<double>(1,1))*wh[K+i] +
                                static_cast<float>(trans2.at<double>(1,2))*1.0;

        target_coords[i*4] = new_pt1.at<float>(0,0);
        target_coords[i*4+1] = new_pt1.at<float>(0,1);
        target_coords[i*4+2] = new_pt2.at<float>(0,0);
        target_coords[i*4+3] = new_pt2.at<float>(0,1);
    }
    
    float alpha;
    float x, y, z, rot_y;
    detected3D.clear();
    for(int i = 0; i<classes; i++){      
        for(int j=0; j<K; j++){
            if(clses[j] == i){
                //get alpha
                if(rot[1*K + j] > rot[5*K + j])
                    alpha = std::atan2(rot[2*K + j], rot[3*K + j]) -0.5 * M_PI;
                else
                    alpha = std::atan2(rot[6*K + j], rot[7*K + j]) +0.5 * M_PI;  
                
                // unproject_2d_to_3d
                z = dep[j] - calibs[bi].at<float>(2,3);// z = depth - P[2, 3]
                x = (target_coords[j*4] * dep[j] - calibs[bi].at<float>(0,3) - calibs[bi].at<float>(0,2) * z) / calibs[bi].at<float>(0,0);
                y = (target_coords[j*4+1] * dep[j] - calibs[bi].at<float>(1,3) - calibs[bi].at<float>(1,2) * z) / calibs[bi].at<float>(1,1) + (dim_[j] / 2);
                // alpha2rot_y
                rot_y = (alpha + std::atan2(target_coords[j*4] - calibs[bi].at<float>(0,2), calibs[bi].at<float>(0,0)));
                if(rot_y>M_PI)
                    rot_y -= 2*M_PI;
                if(rot_y<M_PI)
                    rot_y += 2*M_PI;   

                if(scores[j] > confThreshold) {
                    if(z>0) {
                        // compute_box_3d
                        r.at<float>(0,0) = std::cos(rot_y);
                        r.at<float>(0,2) = std::sin(rot_y);
                        r.at<float>(2,0) = -std::sin(rot_y);
                        r.at<float>(2,2) = std::cos(rot_y);

                        corners.at<float>(0,0) = dim_[2*K+j]/2;
                        corners.at<float>(0,1) = dim_[2*K+j]/2;
                        corners.at<float>(0,2) = -dim_[2*K+j]/2;
                        corners.at<float>(0,3) = -dim_[2*K+j]/2;
                        corners.at<float>(0,4) = dim_[2*K+j]/2;
                        corners.at<float>(0,5) = dim_[2*K+j]/2;
                        corners.at<float>(0,6) = -dim_[2*K+j]/2;
                        corners.at<float>(0,7) = -dim_[2*K+j]/2;

                        corners.at<float>(1,4) = -dim_[j];
                        corners.at<float>(1,5) = -dim_[j];
                        corners.at<float>(1,6) = -dim_[j];
                        corners.at<float>(1,7) = -dim_[j];
                        
                        corners.at<float>(2,0) = dim_[K+j]/2;
                        corners.at<float>(2,1) = -dim_[K+j]/2;
                        corners.at<float>(2,2) = -dim_[K+j]/2;
                        corners.at<float>(2,3) = dim_[K+j]/2;
                        corners.at<float>(2,4) = dim_[K+j]/2;
                        corners.at<float>(2,5) = -dim_[K+j]/2;
                        corners.at<float>(2,6) = -dim_[K+j]/2;
                        corners.at<float>(2,7) = dim_[K+j]/2;
                        cv::Mat aus = r * corners;

                        for(int k=0; k<8; k++) {
                            aus.at<float>(0,k) += x;
                            aus.at<float>(1,k) += y;
                            aus.at<float>(2,k) += z;
                        }
                        // corners.copyTo(pts3DHomo(cv::Rect(0, 0, 8, 3)));
                        for(int k1=0; k1<3; k1++) {
                            for(int k2=0; k2<8; k2++)
                                pts3DHomo.at<float>(k1,k2) = aus.at<float>(k1,k2); 
                        }
                        aus.release();
                        aus = calibs[bi] * pts3DHomo;
                        
                        tk::dnn::box3D res;
                        for(int k=0; k<8; k++) {
                            res.corners.push_back(aus.at<float>(0,k) / aus.at<float>(2,k));
                            res.corners.push_back(aus.at<float>(1,k) / aus.at<float>(2,k));
                        }
                        res.cl = i;
                        res.prob = scores[j];
                        //res.print();
                        detected3D.push_back(res);  
                    }
                }
            }
        }
    }
    batchDetected.push_back(detected3D);
}

void CenternetDetection3D::draw(std::vector<cv::Mat>& frames) {
    tk::dnn::box3D b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;

    int baseline = 0;
    float font_scale = 0.5;
    int thickness = 2;   
    
    for(int bi=0; bi<frames.size(); ++bi){
        float scale_x = float(originalSize[bi].width)/dim.w;
        float scale_y = float(originalSize[bi].height)/dim.h;
        resize(frames[bi], frames[bi], originalSize[bi]);
        // draw dets
        for(int i=0; i<batchDetected[bi].size(); i++) {
            b = batchDetected[bi][i];

            for(int ind_f = 3; ind_f>=0; ind_f--) {
                for(int j=0; j<4; j++) {
                    cv::line(frames[bi], cv::Point(b.corners.at(faceId.at(ind_f).at(j) * 2) * scale_x, 
                                            b.corners.at(faceId.at(ind_f).at(j) * 2 + 1) * scale_y),
                                    cv::Point(b.corners.at(faceId.at(ind_f).at((j+1)%4) * 2) * scale_x, 
                                            b.corners.at(faceId.at(ind_f).at((j+1)%4) * 2 + 1) * scale_y), 
                                    colors[b.cl], 2);
                    if(ind_f == 0) {
                        cv::line(frames[bi], cv::Point(b.corners.at(faceId.at(ind_f).at(0) * 2)  * scale_x, 
                                                b.corners.at(faceId.at(ind_f).at(0) * 2 + 1)* scale_y),
                                        cv::Point(b.corners.at(faceId.at(ind_f).at(2) * 2)  * scale_x, 
                                                b.corners.at(faceId.at(ind_f).at(2) * 2 + 1) * scale_y), colors[b.cl], 2);
                        cv::line(frames[bi], cv::Point(b.corners.at(faceId.at(ind_f).at(1) * 2)* scale_x, 
                                                b.corners.at(faceId.at(ind_f).at(1) * 2 + 1)* scale_y),
                                        cv::Point(b.corners.at(faceId.at(ind_f).at(3) * 2)* scale_x, 
                                                b.corners.at(faceId.at(ind_f).at(3) * 2 + 1)* scale_y), colors[b.cl], 2);
                    }
                }
            }
            // draw label
            cv::Size text_size = getTextSize(classesNames[b.cl], cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
            cv::rectangle(frames[bi],  cv::Point(b.corners.at(faceId.at(0).at(0) * 2)* scale_x, 
                                            b.corners.at(faceId.at(0).at(0) * 2 + 1)* scale_y), 
                                cv::Point((b.corners.at(faceId.at(0).at(0) * 2)* scale_x + text_size.width - 2), 
                                            (b.corners.at(faceId.at(0).at(0) * 2 + 1)* scale_y - text_size.height - 2)), colors[b.cl], -1);                      
            cv::putText(frames[bi], classesNames[b.cl], cv::Point(b.corners.at(faceId.at(0).at(0) * 2)* scale_x,  
                                                            (b.corners.at(faceId.at(0).at(0) * 2 + 1)* scale_y - (baseline / 2))), 
                                                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
        }
    }
}

}}


