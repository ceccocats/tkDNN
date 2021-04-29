#include "CenternetDetection3DTrack.h"


namespace tk { namespace dnn {


bool CenternetDetection3DTrack::init(const std::string& tensor_path, const int n_classes, const int n_batches, const float conf_thresh) {
    netRT = new tk::dnn::NetworkRT(NULL, (tensor_path).c_str() );
    
    dim = netRT->input_dim;
    dim.c = 3;
    nBatches = n_batches;
    confThreshold = conf_thresh;

    init_preprocessing();
    init_pre_inf();
    init_postprocessing();
    init_visualization(n_classes);
    
    count_tr = 0;
}

bool CenternetDetection3DTrack::init_preprocessing(){
    //image transformation
    src = cv::Mat(cv::Size(2,3), CV_32F);
    dst = cv::Mat(cv::Size(2,3), CV_32F);
    dst2 = cv::Mat(cv::Size(2,3), CV_32F);
    trans = cv::Mat(cv::Size(3,2), CV_32F);
    trans2 = cv::Mat(cv::Size(3,2), CV_32F);
    trans_out = cv::Mat(cv::Size(3,2), CV_32F);

    dst2.at<float>(0,0)=width * 0.5;
    dst2.at<float>(0,1)=width * 0.5;
    dst2.at<float>(1,0)=width * 0.5;
    dst2.at<float>(1,1)=width * 0.5 +  width * -0.5; 
    
    dst2.at<float>(2,0)=dst2.at<float>(1,0) + (-dst2.at<float>(0,1)+dst2.at<float>(1,1) );
    dst2.at<float>(2,1)=dst2.at<float>(1,1) + (dst2.at<float>(0,0)-dst2.at<float>(1,0) );


#ifdef OPENCV_CUDACONTRIB

    checkCuda( cudaMalloc(&mean_d, 3 * sizeof(float)) );
    checkCuda( cudaMalloc(&stddev_d, 3 * sizeof(float)) );
    float mean[3] = {0.40789655, 0.44719303, 0.47026116};
    float stddev[3] = {0.2886383, 0.27408165, 0.27809834};
    
    checkCuda(cudaMemcpy(mean_d, mean, 3*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(stddev_d, stddev, 3*sizeof(float), cudaMemcpyHostToDevice));
#else
    checkCuda(cudaMallocHost(&input, sizeof(dnnType)*dim.tot() * nBatches));
    mean << 0.40789655, 0.44719303, 0.47026116;
    stddev << 0.2886383, 0.27408165, 0.27809834;
    
#endif

    checkCuda(cudaMalloc(&input_d, sizeof(dnnType)*netRT->input_dim.tot() * nBatches));
    checkCuda(cudaMalloc(&input_pre_inf_d, sizeof(dnnType)*dim.tot()));
    checkCuda( cudaMalloc(&d_ptrs, dim.tot() * sizeof(float)) );
}

bool CenternetDetection3DTrack::init_pre_inf(){
    // initial steps: the first part of the network
    const char *pre_img_conv1_bin = "dla34_cnet3d_track/layers/base-pre_img_layer-0.bin";
    const char *pre_hm_conv1_bin = "dla34_cnet3d_track/layers/base-pre_hm_layer-0.bin";
    const char *conv1_bin = "dla34_cnet3d_track/layers/base-base_layer-0.bin";
    const char *conv2_bin = "dla34_cnet3d_track/layers/base-level0-0.bin";
    dim_in0 = tk::dnn::dataDim_t(1, 3, 512, 512, 1);
    dim_in1 = tk::dnn::dataDim_t(1, 1, 512, 512, 1);
    

    checkCuda( cudaMalloc(&out_d, netRT->input_dim.tot()*sizeof(dnnType)) );
    checkCuda( cudaMalloc(&img_d, dim_in0.tot()*sizeof(dnnType)) );
    checkCuda( cudaMalloc(&hm_d, dim_in1.tot()*sizeof(dnnType)) );
    // init to zeros hm
     dnnType *hm_h;
    checkCuda( cudaMallocHost(&hm_h, 1 * dim.h * dim.w*sizeof(dnnType)) ); 
    for(int i=0; i<1 * dim.h * dim.w; i++)
        hm_h[i]=0.0f;
    checkCuda( cudaMemcpy(hm_d, hm_h, 1 * dim.h * dim.w * sizeof(dnnType), cudaMemcpyHostToDevice) );
    checkCuda( cudaFreeHost(hm_h) );
    dnnType *i0_h, *i1_h, *i2_h;
    // dnnType *i0_d, *i1_d, *i2_d;

    // const char *input_bin = "dla34_cnet3d_track/debug/input.bin";
    // const char *pre_img_bin = "dla34_cnet3d_track/debug/pre_imgages.bin";
    // const char *pre_hm_bin = "dla34_cnet3d_track/debug/pre_hms.bin";
    // readBinaryFile(pre_img_bin, dim_in0.tot(), &i0_h, &img_d);
    // readBinaryFile(pre_hm_bin, dim_in1.tot(), &i1_h, &hm_d);
    // readBinaryFile(input_bin, dim_in0.tot(), &i2_h, &input_pre_inf_d);

    pre_phase_net = new tk::dnn::Network(dim_in0);
    //pre-img
    tk::dnn::Input *in_pre_img = new tk::dnn::Input(pre_phase_net, dim_in0, img_d);
    tk::dnn::Conv2d *pre_img_conv1 = new tk::dnn::Conv2d(pre_phase_net, 16, 7, 7, 1, 1, 3, 3, pre_img_conv1_bin, true);
    tk::dnn::Activation *pre_img_relu = new tk::dnn::Activation(pre_phase_net, CUDNN_ACTIVATION_RELU);
    //pre-hm
    tk::dnn::Input *in_pre_hm = new tk::dnn::Input(pre_phase_net, dim_in1, hm_d);
    tk::dnn::Conv2d *pre_hm_conv1 = new tk::dnn::Conv2d(pre_phase_net, 16, 7, 7, 1, 1, 3, 3, pre_hm_conv1_bin, true);
    tk::dnn::Activation *pre_hm_relu = new tk::dnn::Activation(pre_phase_net, CUDNN_ACTIVATION_RELU);
    // image input
    tk::dnn::Input *input_image = new tk::dnn::Input(pre_phase_net, dim_in0, input_pre_inf_d);
    tk::dnn::Conv2d *conv1 = new tk::dnn::Conv2d(pre_phase_net, 16, 7, 7, 1, 1, 3, 3, conv1_bin, true);
    tk::dnn::Activation *relu1 = new tk::dnn::Activation(pre_phase_net, CUDNN_ACTIVATION_RELU);
   
    tk::dnn::Shortcut *s0_input = new tk::dnn::Shortcut(pre_phase_net, pre_img_relu);
    tk::dnn::Shortcut *s1_input = new tk::dnn::Shortcut(pre_phase_net, pre_hm_relu);
    // output data
    out_d = s1_input->dstData;
    //print network model
    pre_phase_net->print();

    iter0=true; // in the first iteration the last input is equal to the current input.
    return true;
}

bool CenternetDetection3DTrack::init_postprocessing(){
    srand(0); //seed = 0 for random colors

    dim_hm = tk::dnn::dataDim_t(1, 10, 128, 128, 1);
    dim_wh = tk::dnn::dataDim_t(1, 2, 128, 128, 1);
    dim_reg = tk::dnn::dataDim_t(1, 2, 128, 128, 1);
    dim_track = tk::dnn::dataDim_t(1, 2, 128, 128, 1);
    dim_dep = tk::dnn::dataDim_t(1, 1, 128, 128, 1);
    dim_rot = tk::dnn::dataDim_t(1, 8, 128, 128, 1);
    dim_dim = tk::dnn::dataDim_t(1, 3, 128, 128, 1);
    dim_amodel_offset = tk::dnn::dataDim_t(1, 2, 128, 128, 1);

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

    checkCuda( cudaMallocHost(&bbx0, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bby0, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bbx1, K * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&bby1, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bbx0_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bby0_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bbx1_d, K * sizeof(float)) ); 
    checkCuda( cudaMalloc(&bby1_d, K * sizeof(float)) );

    checkCuda( cudaMallocHost(&intxs, K  * sizeof(int)) ); 
    checkCuda( cudaMallocHost(&intys, K  * sizeof(int)) ); 

    checkCuda( cudaMallocHost(&track, K * dim_track.c * sizeof(float)) );
    checkCuda( cudaMallocHost(&dep, K * dim_dep.c * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&rot, K * dim_rot.c * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&dim_, K * dim_dim.c * sizeof(float)) ); 
    checkCuda( cudaMallocHost(&wh, K * dim_wh.c * sizeof(float)) );
    checkCuda( cudaMallocHost(&amodel_offset, K * dim_amodel_offset.c * sizeof(float)) );
    checkCuda( cudaMalloc(&track_d, K * dim_track.c * sizeof(float)) ); 
    checkCuda( cudaMalloc(&dep_d, K * dim_dep.c * sizeof(float)) ); 
    checkCuda( cudaMalloc(&rot_d, K * dim_rot.c * sizeof(float)) ); 
    checkCuda( cudaMalloc(&dim_d, K * dim_dim.c * sizeof(float)) ); 
    checkCuda( cudaMalloc(&wh_d, K * dim_wh.c * sizeof(float)) );  
    checkCuda( cudaMalloc(&amodel_offset_d, K * dim_amodel_offset.c * sizeof(float)) ); 

    checkCuda( cudaMallocHost(&target_coords, 4 * K *sizeof(float)) );

    calibs = cv::Mat(cv::Size(4,3), CV_32F);
    calibs.at<float>(0,0) = 633.0;
    calibs.at<float>(0,1) = 0.0;
    calibs.at<float>(0,2) = 0.0; //w/2
    calibs.at<float>(0,3) = 0.0;
    calibs.at<float>(1,0) = 0.0;
    calibs.at<float>(1,1) = 633.0;
    calibs.at<float>(1,2) = 0.0; //h/2
    calibs.at<float>(1,3) = 0.0;
    calibs.at<float>(2,0) = 0.0;
    calibs.at<float>(2,1) = 0.0;
    calibs.at<float>(2,2) = 1.0;
    calibs.at<float>(2,3) = 0.0;

    // Alloc array used in the kernel 
    checkCuda( cudaMalloc(&src_out, K *sizeof(float)) );
    checkCuda( cudaMalloc(&ids_out, K *sizeof(int)) );
}

bool CenternetDetection3DTrack::init_visualization(const int n_classes){
    classes = n_classes;
    // const char *kitti_class_name[] = {
    //         "person", "car", "bicycle"};
    // classesNames = std::vector<std::string>(kitti_class_name, std::end( kitti_class_name));
    
    const char *class_name[] = {"car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", 
                                        "motorcycle", "bicycle", "traffic_cone", "barrier"};
    classesNames = std::vector<std::string>(class_name, std::end( class_name));
    
    // const char *coco_class_name[] = {
    //         "person", "bicycle", "car", "motorcycle", "airplane", 
    //         "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    //         "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    //         "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    //         "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    //         "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    //         "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    //         "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    //         "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    //         "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    //         "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    //         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    //         "scissors", "teddy bear", "hair drier", "toothbrush"
    //         };
    // classesNames = std::vector<std::string>(coco_class_name, std::end( coco_class_name));

    for(int c=0; c<classes; c++) {
        int offset = c*123457 % classes;
        float r = getColor(2, offset, classes);
        float g = getColor(1, offset, classes);
        float b = getColor(0, offset, classes);
        colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
    }
    for(int c=0; c<256; c++) {
        int offset = c*123457 % 256;
        float r = getColor(2, offset, 256);
        float g = getColor(1, offset, 256);
        float b = getColor(0, offset, 256);
        tr_colors[c] = cv::Scalar(int(255.0*b), int(255.0*g), int(255.0*r));
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

    face_id.push_back({0,1,5,4});
    face_id.push_back({1,2,6, 5});
    face_id.push_back({3,0,4,7});
    face_id.push_back({2,3,7,6});
    // ([[0,1,5,4], [1,2,6, 5], [2,3,7,6], [3,0,4,7]]);
}

void CenternetDetection3DTrack::_get_additional_inputs(){
    //None no additional input
}

void CenternetDetection3DTrack::pre_inf(const int bi){
    TKDNN_TSTART
    tk::dnn::dataDim_t dim_aus;
    pre_phase_net->infer(dim_aus, nullptr);
    TKDNN_TSTOP
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaMemcpy(input_d+ netRT->input_dim.tot()*bi, pre_phase_net->layers[pre_phase_net->num_layers-1]->dstData, netRT->input_dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice) );
    checkCuda( cudaDeviceSynchronize() );
}

void CenternetDetection3DTrack::preprocess(cv::Mat &frame, const int bi){
    // -----------------------------------pre-process ------------------------------------------
    batchTracked.clear();
    cv::Size sz = originalSize[bi];
    cv::Size sz_old;
    float scale = 1.0;
    float new_height = sz.height * scale;
    float new_width = sz.width * scale;
    if(sz.height != sz_old.height && sz.width != sz_old.width){
        calibs.at<float>(0,2) = new_width / 2.0f;
        calibs.at<float>(1,2) = new_height /2.0f;
        float c[] = {new_width / 2.0f, new_height /2.0f};
        float s[] = {dim.w, dim.h};
        // float s = new_width >= new_height ? new_width : new_height;
        // ----------- get_affine_transform
        // rot_rad = pi * 0 / 100 --> 0
        //dim.print();
        src.at<float>(0,0)=c[0];
        src.at<float>(0,1)=c[1];
        src.at<float>(1,0)=c[0];
        src.at<float>(1,1)=c[1] + s[0] * -0.5;
        dst.at<float>(0,0)=dim.w * 0.5;
        dst.at<float>(0,1)=dim.h * 0.5;
        dst.at<float>(1,0)=dim.w * 0.5;
        dst.at<float>(1,1)=dim.h * 0.5 +  dim.w * -0.5; 
        
        src.at<float>(2,0)=src.at<float>(1,0) + (-src.at<float>(0,1)+src.at<float>(1,1) );
        src.at<float>(2,1)=src.at<float>(1,1) + (src.at<float>(0,0)-src.at<float>(1,0) );
        dst.at<float>(2,0)=dst.at<float>(1,0) + (-dst.at<float>(0,1)+dst.at<float>(1,1) );
        dst.at<float>(2,1)=dst.at<float>(1,1) + (dst.at<float>(0,0)-dst.at<float>(1,0) );


        trans = cv::getAffineTransform( src, dst );
        trans2 = cv::getAffineTransform( dst2, src );
        trans2.convertTo(trans_out, CV_32F);
    }
    sz_old = sz;
#ifdef OPENCV_CUDACONTRIB
    std::cout<<"OPENCV CPMTROB\n";
    cv::cuda::GpuMat im_Orig; 
    cv::cuda::GpuMat imageF1_d, imageF2_d;
        
    im_Orig = cv::cuda::GpuMat(frame);
    // cv::cuda::resize (im_Orig, imageF1_d, cv::Size(new_width, new_height)); 
    imageF1_d = im_Orig;
    checkCuda( cudaDeviceSynchronize() );
    
    sz = imageF1_d.size();

    cv::cuda::warpAffine(imageF1_d, imageF2_d, trans, cv::Size(dim.w, dim.h), cv::INTER_LINEAR );
    checkCuda( cudaDeviceSynchronize() );
    
    imageF2_d.convertTo(imageF1_d, CV_32FC3, 1/255.0); 
    checkCuda( cudaDeviceSynchronize() );
    
    dim2 = dim;
    cv::cuda::GpuMat bgr[3]; 
    cv::cuda::split(imageF1_d,bgr);//split source

    for(int i=0; i<dim.c; i++)
        checkCuda( cudaMemcpy(d_ptrs + i*dim.h * dim.w, (float*)bgr[i].data, dim.h * dim.w * sizeof(float), cudaMemcpyDeviceToDevice) );
        
    normalize(d_ptrs, dim.c, dim.h, dim.w, mean_d, stddev_d);

    checkCuda(cudaMemcpy(input_pre_inf_d, d_ptrs, dim2.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));
    checkCuda( cudaDeviceSynchronize() );

#else
    std::cout<<"NO OPENCV CPMTROB\n";
    cv::Mat imageF;
    // resize(frame, imageF, cv::Size(new_width, new_height));
    imageF = frame;
    sz = imageF.size();
 
    cv::warpAffine(imageF, imageF, trans, cv::Size(dim.w, dim.h), cv::INTER_LINEAR );
 
    sz = imageF.size();
    imageF.convertTo(imageF, CV_32FC3, 1/255.0); 
 
    dim2 = dim;
    //split channels
    cv::Mat bgr[3]; 
    cv::split(imageF,bgr);//split source
    
    for(int i=0; i<3; i++){
        bgr[i] = bgr[i] - mean[i];
        bgr[i] = bgr[i] / stddev[i];
    }
    for(int i=0; i<dim2.c; i++) {
        int idx = i*imageF.rows*imageF.cols;
        int ch =i;//dim2.c-3 +i;//i;// 
        memcpy((void*)&input[idx], (void*)bgr[ch].data, imageF.rows*imageF.cols*sizeof(dnnType));
    }
    checkCuda(cudaMemcpyAsync(input_pre_inf_d, input, dim2.tot()*sizeof(dnnType), cudaMemcpyHostToDevice));
    checkCuda( cudaDeviceSynchronize() );

#endif

    if(iter0) {
        checkCuda( cudaMemcpy(img_d, input_pre_inf_d, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice) );
        checkCuda( cudaDeviceSynchronize() );
        iter0=false;
    }
    pre_inf(bi);

    checkCuda( cudaMemcpy(img_d, input_pre_inf_d, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice) );
    checkCuda( cudaDeviceSynchronize() );
}

cv::Mat CenternetDetection3DTrack::transform_preds_with_trans(float x1, float x2){
    cv::Mat target_coords(cv::Size(1,3), CV_32F);
    target_coords.at<float>(0,0) = x1;
    target_coords.at<float>(0,1) = x2;
    target_coords.at<float>(0,2) = 1.0;
    return trans_out * target_coords;
}

void CenternetDetection3DTrack::tracking(){
   
    float item_size[count_det];
    int item_cl[count_det];
    float dets[2*count_det];
    for(int i=0; i<count_det; i++){
        item_size[i] = (det_res[i].bb1.at<float>(0,0) - det_res[i].bb0.at<float>(0,0)) * 
                    (det_res[i].bb1.at<float>(0,1) - det_res[i].bb0.at<float>(0,1));
        item_cl[i] = det_res[i].cl;
        dets[i*2] = det_res[i].ct.at<float>(0,0);
        dets[i*2+1] = det_res[i].ct.at<float>(0,1);
    }
    
    float track_size[count_tr];
    int track_cl[count_tr];
    float tracks[2*count_tr];
    for(int i=0; i<count_tr; i++){
        track_size[i] = (tr_res[i].det_res.bb1.at<float>(0,0) - tr_res[i].det_res.bb0.at<float>(0,0)) * 
                        (tr_res[i].det_res.bb1.at<float>(0,1) - tr_res[i].det_res.bb0.at<float>(0,1));
        track_cl[i] = tr_res[i].det_res.cl;
        tracks[i*2] = tr_res[i].det_res.ct.at<float>(0,0);
        tracks[i*2+1] = tr_res[i].det_res.ct.at<float>(0,1);
    } 
    float dist[count_tr*count_det];
    bool invalid;
    for(int i=0; i<count_tr; i++){
        for(int j=0; j<count_det; j++){
            dist[j*count_tr+i] =  pow((tracks[i*2] - dets[j*2]), 2) + 
                            pow((tracks[i*2+1] - dets[j*2+1]), 2);
            invalid = dist[j*count_tr+i] > track_size[i] || dist[j*count_tr+i] > item_size[j] || item_cl[j] != track_cl[i];
            dist[j*count_tr+i] = dist[j*count_tr+i] + invalid * (1 << 18);
        }
    }
    int matched_indices[2*count_tr];
    float min_tr;
    int min_idtr=-1;
    for(int i=0; i<count_tr; i++) {
        matched_indices[i*2] = -1;
        matched_indices[i*2+1] = -1;
    }
    for(int i=0; i<count_det; i++){
        min_tr=(1 << 18);
        for(int j=0; j<count_tr; j++){
            if(dist[i*count_tr+j]<min_tr) {
                min_tr = dist[i*count_tr+j];
                min_idtr = j;
            }
        }
        if(min_tr < (1<<16)) {
            for(int j=0; j<count_det; j++){
                dist[j*count_tr+min_idtr] = (1 << 18);
            }
            matched_indices[2*min_idtr] = min_idtr;
            matched_indices[2*min_idtr+1] = i;
        }
    }
    
    bool unmatched_dets[count_det];
    for(int i=0; i<count_det; i++)
        unmatched_dets[i] = false;
    bool unmatched_tracks[count_tr];
    for(int i=0; i<count_tr; i++) 
        unmatched_tracks[i] = false;
    for(int i=0; i<count_tr; i++) {
        if(matched_indices[2*i] != -1)
            unmatched_tracks[matched_indices[2*i]]=true;
        
        if(matched_indices[2*i+1] != -1)
            unmatched_dets[matched_indices[2*i+1]]=true;        
    }

    //match
    for(int i=0; i<count_tr; i++) {
        if(matched_indices[2*i+1] != -1 && matched_indices[2*i] != -1) { //second condition is optional
            int tr_id = matched_indices[2*i];
            int d_id = matched_indices[2*i+1];
            
            // tr_res[tr_id].det_res = det_res[d_id];
            tr_res[tr_id].det_res.score = det_res[d_id].score;
            tr_res[tr_id].det_res.cl = det_res[d_id].cl;
            tr_res[tr_id].det_res.ct = det_res[d_id].ct;
            tr_res[tr_id].det_res.tr = det_res[d_id].tr;
            tr_res[tr_id].det_res.bb0 = det_res[d_id].bb0;
            tr_res[tr_id].det_res.bb1 = det_res[d_id].bb1;
            tr_res[tr_id].det_res.dep = det_res[d_id].dep;
            tr_res[tr_id].det_res.dim[0] = det_res[d_id].dim[0];
            tr_res[tr_id].det_res.dim[1] = det_res[d_id].dim[1];
            tr_res[tr_id].det_res.dim[2] = det_res[d_id].dim[2];
            tr_res[tr_id].det_res.alpha = det_res[d_id].alpha;
            tr_res[tr_id].det_res.x = det_res[d_id].x;
            tr_res[tr_id].det_res.y = det_res[d_id].y;
            tr_res[tr_id].det_res.z = det_res[d_id].z;
            tr_res[tr_id].det_res.rot_y = det_res[d_id].rot_y;
            // tr_res[matched_indices[2*i]].tracking_id = ; is the same 
            // tr_res[matched_indices[2*i]].color = ; is the same
            tr_res[tr_id].age = 1; 
            tr_res[tr_id].active = tr_res[tr_id].active+1;
        }
    }
    //delete target umatched track
    int new_count_tr = 0;
    for(int i=0; i<count_tr; i++) {
        if(unmatched_tracks[i]) 
            new_count_tr++;
    }
    if(new_count_tr == 0 && count_tr != 0) {        //reset
        tr_res.clear();
        count_tr = 0;
    }

    int old_count_tr = count_tr;
    if(count_tr != 0 && new_count_tr != count_tr) {
        std::vector<struct trackingRes> new_tr_res;
        int id_new_tr=0;
        for(int i=0; i<count_tr; i++) {
            if(unmatched_tracks[i]) {
                struct trackingRes new_tr_res_;
                // new_tr_res_new_det_res.det_res = tr_res[i].det_res;
                new_tr_res_.det_res.score = tr_res[i].det_res.score;
                new_tr_res_.det_res.cl = tr_res[i].det_res.cl;
                new_tr_res_.det_res.ct = tr_res[i].det_res.ct;
                new_tr_res_.det_res.tr = tr_res[i].det_res.tr;
                new_tr_res_.det_res.bb0 = tr_res[i].det_res.bb0;
                new_tr_res_.det_res.bb1 = tr_res[i].det_res.bb1;
                new_tr_res_.det_res.dep = tr_res[i].det_res.dep;
                new_tr_res_.det_res.dim[0] = tr_res[i].det_res.dim[0];
                new_tr_res_.det_res.dim[1] = tr_res[i].det_res.dim[1];
                new_tr_res_.det_res.dim[2] = tr_res[i].det_res.dim[2];
                new_tr_res_.det_res.alpha = tr_res[i].det_res.alpha;
                new_tr_res_.det_res.x = tr_res[i].det_res.x;
                new_tr_res_.det_res.y = tr_res[i].det_res.y;
                new_tr_res_.det_res.z = tr_res[i].det_res.z;
                new_tr_res_.det_res.rot_y = tr_res[i].det_res.rot_y;
                new_tr_res_.tracking_id = tr_res[i].tracking_id;
                new_tr_res_.age = tr_res[i].age;
                new_tr_res_.active = tr_res[i].active;
                new_tr_res_.color = tr_res[i].color;
                id_new_tr++;
                new_tr_res.push_back(new_tr_res_);
            }
        }
        
        if(count_tr) {
            tr_res.clear();
        }
        count_tr = new_count_tr;
        tr_res=new_tr_res;
    }
    
    int count_tr_ = count_tr;
    for(int i=0; i<count_det; i++) {
        if((!unmatched_dets[i]) && det_res[i].score > new_thresh) {
            count_tr_ ++;
            struct trackingRes new_tr_res_;
            new_tr_res_.det_res.score = det_res[i].score;
            new_tr_res_.det_res.cl = det_res[i].cl;
            new_tr_res_.det_res.ct = det_res[i].ct;
            new_tr_res_.det_res.tr = det_res[i].tr;
            new_tr_res_.det_res.bb0 = det_res[i].bb0;
            new_tr_res_.det_res.bb1 = det_res[i].bb1;
            new_tr_res_.det_res.dep = det_res[i].dep;
            new_tr_res_.det_res.dim[0] = det_res[i].dim[0];
            new_tr_res_.det_res.dim[1] = det_res[i].dim[1];
            new_tr_res_.det_res.dim[2] = det_res[i].dim[2];
            new_tr_res_.det_res.alpha = det_res[i].alpha;
            new_tr_res_.det_res.x = det_res[i].x;
            new_tr_res_.det_res.y = det_res[i].y;
            new_tr_res_.det_res.z = det_res[i].z;
            new_tr_res_.det_res.rot_y = det_res[i].rot_y;
            new_tr_res_.tracking_id = track_id++; 
            new_tr_res_.age = 1; 
            new_tr_res_.active = 1;
            new_tr_res_.color = rand() % 256;
            tr_res.push_back(new_tr_res_);
        }
    }
    count_tr = count_tr_;
    
    if(track_id==1000)
        track_id=0;
    det_res.clear();
    
}

void CenternetDetection3DTrack::postprocess(const int bi, const bool mAP) {
    dnnType *rt_out[9];
    rt_out[0] = (dnnType *)netRT->buffersRT[1]+ netRT->buffersDIM[1].tot()*bi;
    rt_out[1] = (dnnType *)netRT->buffersRT[2]+ netRT->buffersDIM[2].tot()*bi;
    rt_out[2] = (dnnType *)netRT->buffersRT[3]+ netRT->buffersDIM[3].tot()*bi; 
    rt_out[3] = (dnnType *)netRT->buffersRT[4]+ netRT->buffersDIM[4].tot()*bi; 
    rt_out[4] = (dnnType *)netRT->buffersRT[5]+ netRT->buffersDIM[5].tot()*bi; 
    rt_out[5] = (dnnType *)netRT->buffersRT[6]+ netRT->buffersDIM[6].tot()*bi; 
    rt_out[6] = (dnnType *)netRT->buffersRT[7]+ netRT->buffersDIM[7].tot()*bi; 
    rt_out[7] = (dnnType *)netRT->buffersRT[8]+ netRT->buffersDIM[8].tot()*bi; 
    rt_out[8] = (dnnType *)netRT->buffersRT[9]+ netRT->buffersDIM[9].tot()*bi; 
    
    // ------------------------------------ process --------------------------------------------
    
    activationSIGMOIDForward(rt_out[0], rt_out[0], dim_hm.tot());
    checkCuda( cudaDeviceSynchronize() );    

    // output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    activationSIGMOIDForward(rt_out[5], rt_out[5], dim_dep.tot());
    checkCuda( cudaDeviceSynchronize() );
    transformDep(ones, ones + dim_dep.tot(), rt_out[5], rt_out[5] + dim_dep.tot());
    checkCuda( cudaDeviceSynchronize() );

    // nms 
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
    checkCuda( cudaDeviceSynchronize() ); 
    checkCuda( cudaMemcpy(topk_xs_d, (float *)inttopk_xs_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    checkCuda( cudaMemcpy(topk_ys_d, (float *)inttopk_ys_d, K*sizeof(float), cudaMemcpyDeviceToDevice) );
    
    checkCuda( cudaMemcpy(intxs, inttopk_xs_d, K * sizeof(int), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(intys, inttopk_ys_d, K * sizeof(int), cudaMemcpyDeviceToHost) ); 
    
    checkCuda( cudaMemcpy(clses, clses_d, K*sizeof(int), cudaMemcpyDeviceToHost) );
    
    // ----------- topk end 
    
    topKxyAddOffset(topk_inds_d, K, dim_reg.h*dim_reg.w, inttopk_xs_d, inttopk_ys_d, topk_xs_d, topk_ys_d, rt_out[3], src_out, ids_out);
    checkCuda( cudaDeviceSynchronize() );

    bboxes(topk_inds_d, K, dim_wh.h*dim_wh.w, topk_xs_d, topk_ys_d, rt_out[2], bbx0_d, bbx1_d, bby0_d, bby1_d, src_out, ids_out);
    checkCuda( cudaDeviceSynchronize() );
    checkCuda( cudaMemcpy(bbx0, bbx0_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bby0, bby0_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bbx1, bbx1_d, K * sizeof(float), cudaMemcpyDeviceToHost) ); 
    checkCuda( cudaMemcpy(bby1, bby1_d, K * sizeof(float), cudaMemcpyDeviceToHost) );

    //regression heads
    // ['tracking', 'dep', 'rot', 'dim', 'amodel_offset',
    // 'nuscenes_att', 'velocity']
    getRecordsFromTopKId(topk_inds_d, K, dim_track.c, dim_track.h * dim_track.w, rt_out[4], track_d, ids_out);
    checkCuda( cudaMemcpy(track, track_d, K * dim_track.c * sizeof(float), cudaMemcpyDeviceToHost) );

    getRecordsFromTopKId(topk_inds_d, K, dim_dep.c, dim_dep.h * dim_dep.w, rt_out[5], dep_d, ids_out);
    checkCuda( cudaMemcpy(dep, dep_d, K * dim_dep.c * sizeof(float), cudaMemcpyDeviceToHost) );
    
    getRecordsFromTopKId(topk_inds_d, K, dim_rot.c, dim_rot.h * dim_rot.w, rt_out[6], rot_d, ids_out);
    checkCuda( cudaMemcpy(rot, rot_d, K * dim_rot.c * sizeof(float), cudaMemcpyDeviceToHost) );
    
    getRecordsFromTopKId(topk_inds_d, K, dim_dim.c, dim_dim.h * dim_dim.w, rt_out[7], dim_d, ids_out);
    checkCuda( cudaMemcpy(dim_, dim_d, K * dim_dim.c * sizeof(float), cudaMemcpyDeviceToHost) );

    getRecordsFromTopKId(topk_inds_d, K, dim_amodel_offset.c, dim_amodel_offset.h * dim_amodel_offset.w, rt_out[8], amodel_offset_d, ids_out);
    checkCuda( cudaMemcpy(amodel_offset, amodel_offset_d, K * dim_amodel_offset.c * sizeof(float), cudaMemcpyDeviceToHost) );    
    
    // ---------------------------------- post-process -----------------------------------------
    
    count_det = 0;
    det_res.clear();
    for(int i = 0; i<K; i++){
        if(scores[i] < out_thresh) 
            break;
        
        count_det ++;
        struct detectionRes new_det_res;
        new_det_res.score = scores[i]; 
        new_det_res.cl = clses[i]+1; 
        // ret_s=scores[i];
        // ret_c=clses[i]+1;
        new_det_res.ct = transform_preds_with_trans(intxs[i], intys[i]);
        new_det_res.tr = transform_preds_with_trans(intxs[i] + track[i], intys[i] + track[i+K]);
        new_det_res.tr = new_det_res.tr -new_det_res.ct;
        new_det_res.bb0 = transform_preds_with_trans(bbx0[i], bby0[i]);
        new_det_res.bb1 = transform_preds_with_trans(bbx1[i], bby1[i]);
        
        new_det_res.ct = transform_preds_with_trans(((bbx0[i]+bbx1[i])/2 + amodel_offset[i]), 
                                        ((bby0[i]+bby1[i])/2 + amodel_offset[i+K]));
        new_det_res.dep = dep[i]; 
        new_det_res.dim[0] = dim_[i];
        new_det_res.dim[1] = dim_[i+K];
        new_det_res.dim[2] = dim_[i+2*K];
        
        // unproject_2d_to_3d
        new_det_res.z = dep[i] - calibs.at<float>(2,3);
        new_det_res.x = ((float)new_det_res.ct.at<float>(0,0) * dep[i] - calibs.at<float>(0,3) - calibs.at<float>(0,2) * new_det_res.z) / calibs.at<float>(0,0);
        new_det_res.y = ((float)new_det_res.ct.at<float>(0,1) * dep[i] - calibs.at<float>(1,3) - calibs.at<float>(1,2) * new_det_res.z) / calibs.at<float>(1,1) + (dim_[i] / 2);
        
        // alpha2rot_y
        // idx = rot[:, 1] > rot[:, 5]
        // alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
        // alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + ( 0.5 * np.pi)
        // return alpha1 * idx + alpha2 * (1 - idx)
        if(rot[1*K + i] > rot[5*K + i])
            new_det_res.alpha = std::atan2(rot[2*K + i], rot[3*K + i]) -0.5 * M_PI;
        else
            new_det_res.alpha = std::atan2(rot[6*K + i], rot[7*K + i]) +0.5 * M_PI;
        new_det_res.rot_y = (new_det_res.alpha + std::atan2((float)new_det_res.ct.at<float>(0,0) - calibs.at<float>(0,2), calibs.at<float>(0,0)));
        new_det_res.ct = new_det_res.ct + new_det_res.tr;   //dest  
        det_res.push_back(new_det_res);    

    }    
    // track step
    tracking();
    batchTracked.push_back(tr_res);
}

void CenternetDetection3DTrack::draw(std::vector<cv::Mat>& frames) {
    struct trackingRes t;
    float sc;
    int id;
    std::string txt;
    int baseline = 0;
    float font_scale = 0.8;
    int thickness = 2;
    for(int bi=0; bi<frames.size(); ++bi) {
        // draw dets
        for(int i=0; i<batchTracked[bi].size(); i++) {
            t = batchTracked[bi][i];
            id = t.tracking_id;
            txt = classesNames[t.det_res.cl-1]+'-'+std::to_string(id); //forse ha bisogno di cl-1
            cv::Size text_size = getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
            
            if(t.det_res.score > vis_thresh){// && t.active!=0) {
                if(view2d) {
                    cv::rectangle(frames[bi],  cv::Point(t.det_res.bb0.at<float>(0,0), t.det_res.bb0.at<float>(0,1)), 
                                        cv::Point(t.det_res.bb1.at<float>(0,0), t.det_res.bb1.at<float>(0,1)), tr_colors[t.color], thickness);                      
                    cv::rectangle(frames[bi],  cv::Point(t.det_res.bb0.at<float>(0,0), 
                                                    t.det_res.bb0.at<float>(0,1) - text_size.height - thickness), 
                                        cv::Point(t.det_res.bb0.at<float>(0,0) + text_size.width, 
                                                    t.det_res.bb0.at<float>(0,1)), tr_colors[t.color], -1);                      
                                        
                    cv::putText(frames[bi], txt, cv::Point(t.det_res.bb0.at<float>(0,0),  
                                                    t.det_res.bb0.at<float>(0,1) - thickness -1), 
                                                        cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), 1);

                    cv::arrowedLine(frames[bi], cv::Point((int)t.det_res.ct.at<float>(0,0), 
                                                    (int)t.det_res.ct.at<float>(0,1)), 
                                            cv::Point((int)(t.det_res.ct.at<float>(0,0) + t.det_res.tr.at<float>(0,0)),
                                                    (int)(t.det_res.ct.at<float>(0,1) + t.det_res.tr.at<float>(0,1))),
                                            cv::Scalar(255, 0, 255), 2);
                }
                //3d
                if(!view2d && t.det_res.z > 1){
                    r.at<float>(0,0) = std::cos(t.det_res.rot_y);
                    r.at<float>(0,2) = std::sin(t.det_res.rot_y);
                    r.at<float>(2,0) = -std::sin(t.det_res.rot_y);
                    r.at<float>(2,2) = std::cos(t.det_res.rot_y);

                    corners.at<float>(0,0) = t.det_res.dim[2]/2;
                    corners.at<float>(0,1) = t.det_res.dim[2]/2;
                    corners.at<float>(0,2) = -t.det_res.dim[2]/2;
                    corners.at<float>(0,3) = -t.det_res.dim[2]/2;
                    corners.at<float>(0,4) = t.det_res.dim[2]/2;
                    corners.at<float>(0,5) = t.det_res.dim[2]/2;
                    corners.at<float>(0,6) = -t.det_res.dim[2]/2;
                    corners.at<float>(0,7) = -t.det_res.dim[2]/2;

                    corners.at<float>(1,4) = -t.det_res.dim[0];
                    corners.at<float>(1,5) = -t.det_res.dim[0];
                    corners.at<float>(1,6) = -t.det_res.dim[0];
                    corners.at<float>(1,7) = -t.det_res.dim[0];
                    
                    corners.at<float>(2,0) = t.det_res.dim[1]/2;
                    corners.at<float>(2,1) = -t.det_res.dim[1]/2;
                    corners.at<float>(2,2) = -t.det_res.dim[1]/2;
                    corners.at<float>(2,3) = t.det_res.dim[1]/2;
                    corners.at<float>(2,4) = t.det_res.dim[1]/2;
                    corners.at<float>(2,5) = -t.det_res.dim[1]/2;
                    corners.at<float>(2,6) = -t.det_res.dim[1]/2;
                    corners.at<float>(2,7) = t.det_res.dim[1]/2;
                    
                    cv::Mat aus = r * corners;

                    for(int k=0; k<8; k++) {
                        aus.at<float>(0,k) += t.det_res.x;
                        aus.at<float>(1,k) += t.det_res.y;
                        aus.at<float>(2,k) += t.det_res.z;
                    }
                    
                    // corners.copyTo(pts3DHomo(cv::Rect(0, 0, 8, 3)));
                    for(int k1=0; k1<3; k1++) {
                        for(int k2=0; k2<8; k2++)
                            pts3DHomo.at<float>(k1,k2) = aus.at<float>(k1,k2); 
                    }
                    
                    aus.release();
                    aus = calibs * pts3DHomo;
                    std::vector<float> res_corners;
                    for(int k=0; k<8; k++) {
                        res_corners.push_back(aus.at<float>(0,k) / aus.at<float>(2,k));
                        res_corners.push_back(aus.at<float>(1,k) / aus.at<float>(2,k));
                    }
                    aus.release();
                    for(int ind_f = 3; ind_f>=0; ind_f--) {
                        for(int j=0; j<4; j++) {
                            cv::line(frames[bi], cv::Point((int)res_corners.at(face_id.at(ind_f).at(j) * 2), 
                                                    (int)res_corners.at(face_id.at(ind_f).at(j) * 2 + 1)),
                                            cv::Point((int)res_corners.at(face_id.at(ind_f).at((j+1)%4) * 2), 
                                                    (int)res_corners.at(face_id.at(ind_f).at((j+1)%4) * 2 + 1)), 
                                            tr_colors[t.color], 2);
                            if(ind_f == 0 && j==3) {
                                cv::line(frames[bi], cv::Point((int)res_corners.at(face_id.at(ind_f).at(0) * 2), 
                                                        (int)res_corners.at(face_id.at(ind_f).at(0) * 2 + 1)),
                                                cv::Point((int)res_corners.at(face_id.at(ind_f).at(2) * 2), 
                                                        (int)res_corners.at(face_id.at(ind_f).at(2) * 2 + 1)), tr_colors[t.color], 2);
                                cv::line(frames[bi], cv::Point((int)res_corners.at(face_id.at(ind_f).at(1) * 2), 
                                                        (int)res_corners.at(face_id.at(ind_f).at(1) * 2 + 1)),
                                                cv::Point((int)res_corners.at(face_id.at(ind_f).at(3) * 2), 
                                                        (int)res_corners.at(face_id.at(ind_f).at(3) * 2 + 1)), tr_colors[t.color], 2);
                            }
                        }
                    }
                    float bb0=(1 << 10), bb1=0, bb2=(1 << 10), bb3=0;
                    for(int k=0; k<8; k++) {
                        if(res_corners[2*k]<bb0)
                            bb0=res_corners[2*k];
                        if(res_corners[2*k]>bb1)
                            bb1=res_corners[2*k];
                        if(res_corners[2*k+1]<bb2)
                            bb2=res_corners[2*k+1];
                        if(res_corners[2*k+1]>bb3)
                            bb3=res_corners[2*k+1];
                                
                    }
                    // if(not no_bbox):
                    // cv::rectangle(frame,  cv::Point(bb0, bb2), cv::Point(bb1, bb3), 
                    //                                 tr_colors[t.color], thickness);                      
                    cv::rectangle(frames[bi],  cv::Point(bb0, bb2 - text_size.height - thickness), 
                                        cv::Point(bb0 + text_size.width, bb2), tr_colors[t.color], -1);                      
                                        
                    cv::putText(frames[bi], txt, cv::Point(bb0, bb2 - thickness -1), cv::FONT_HERSHEY_SIMPLEX, 
                                                    font_scale, cv::Scalar(255, 255, 255), 1);

                    cv::arrowedLine(frames[bi], cv::Point((int)((bb0 + bb1)/2), (int)((bb2 + bb3)/2)), 
                                        cv::Point((int)((bb0 + bb1)/2 + t.det_res.tr.at<float>(0,0)),
                                                (int)((bb2 + bb3)/2 + t.det_res.tr.at<float>(0,1))),
                                        cv::Scalar(255, 0, 255), 2);
                }
            }
        }
    }
}

}}


