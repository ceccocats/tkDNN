#include<iostream>
#include "tkdnn.h"
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define VOC

#ifdef VOC
const char *reg_bias = "../tests/yolo_voc/layers/g31.bin";
#define CLASS 20
#else
const char *reg_bias = "../tests/yolo/layers/g31.bin";
#define CLASS 80
#endif

int prob_sort(const void *pa, const void *pb) {
    tkDNN::box a = *(tkDNN::box *)pa;
    tkDNN::box b = *(tkDNN::box *)pb;
    float diff = a.prob - b.prob;
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

cv::Mat GetSquareImage(const cv::Mat& img, int target_width) {
    int width = img.cols, height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}

//return inference time
double compute_image( cv::Mat imageORIG, 
                    tkDNN::NetworkRT *netRT, tkDNN::RegionInterpret *rI,
                    dnnType *input, dnnType *output) {
    TIMER_START

    //Resize with padding and convert to float
    cv::Mat image = GetSquareImage(imageORIG, netRT->input_dim.w);
    cv::Mat imageF;
    image.convertTo(imageF, CV_32FC3, 1/255.0); 

    //split channels
    cv::Mat bgr[3];   //destination array
    cv::split(imageF,bgr);//split source
    
    //write channels
    int idx = 0;
    memcpy((void*)&input[idx], (void*)bgr[2].data, imageF.rows*imageF.cols*sizeof(dnnType));
    idx = imageF.rows*imageF.cols;
    memcpy((void*)&input[idx], (void*)bgr[1].data, imageF.rows*imageF.cols*sizeof(dnnType));
    idx *= 2;    
    memcpy((void*)&input[idx], (void*)bgr[0].data, imageF.rows*imageF.cols*sizeof(dnnType));

    //DO INFERENCE
    checkCuda(  cudaMemcpyAsync(netRT->buffersRT[netRT->buf_input_idx], input, 
                                netRT->input_dim.tot()*sizeof(float), 
                                cudaMemcpyHostToDevice, netRT->stream));
    netRT->enqueue();
    checkCuda(  cudaMemcpyAsync(output, netRT->buffersRT[netRT->buf_output_idx], 
                                netRT->output_dim.tot()*sizeof(float), 
                                cudaMemcpyDeviceToHost, netRT->stream));
    cudaStreamSynchronize(netRT->stream);
    
    

    rI->interpretData(output, imageORIG.cols, imageORIG.rows);

    TIMER_STOP
    return t_ns;
}



int print_usage() {
    std::cout<<"usage: ./live net.rt camera_idx\n";
    return 1;
}




int main(int argc, char *argv[]) {

    //params
    char *tensor_path = NULL;
    int   device      = 0;
    float thresh = 0.3f;
    bool  show = false;

    //parse params
    int c;
    while ((c = getopt (argc, argv, "t:si:")) != -1) {
        switch(c) {
        case 't':   thresh = atof(optarg);       break;
        case 's':   show = true;                 break;
        case '?':   
            return print_usage();
        default:    return print_usage();
        }
    }

    if(argc - optind == 2) {
        tensor_path   = argv[optind];
        device        = atoi(argv[optind+1]);
    } else {
        std::cout<<"not enough arguments.\n";
        return print_usage();
    }
    //end parsing

    //std::cout<<"open video stream on device: "<<device<<"\n";
    //cv::VideoCapture cap(device);
		const char* pipe = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
		/*const char* pipe = "nvcamerasrc ! "
		"video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, framerate=(fraction)30/1 ! "
		"nvvidconv ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
		"videoconvert ! appsink";*/

		cv::VideoCapture cap(pipe);

    if(!cap.isOpened())
        FatalError("unable to open video stream");
    //cap.set(CV_CAP_PROP_BUFFERSIZE, 1); // process only last frame

    if(!fileExist(tensor_path))
        FatalError("unable to read serialRT file");

    //convert network to tensorRT
    tkDNN::NetworkRT netRT(NULL, tensor_path);
    tkDNN::RegionInterpret rI(netRT.input_dim, netRT.output_dim, CLASS, 4, 5, thresh, reg_bias);

    dnnType *input = new float[netRT.input_dim.tot()];
    dnnType *output = new float[netRT.output_dim.tot()];

    double mTime = 0;
    int processed_images = 0;


    for(;;) {
        
        //LOAD IMAGE
        cv::Mat img; //= cv::imread("../demo/live/test.jpeg", CV_LOAD_IMAGE_COLOR);
        cap >> img;   

	if(!img.data)                              
            FatalError("Could not open image");
        std::cout<<"Image size: ("<<img.cols<<"x"<<img.rows<<")\n";

        mTime += compute_image(img, &netRT, &rI, input, output); 

        qsort(rI.res_boxes, rI.res_boxes_n, sizeof(tkDNN::box), prob_sort);
        for(int i=0; i<rI.res_boxes_n; i++) {
            tkDNN::box bx = rI.res_boxes[i];
            std::cout<<" ("<<int(bx.prob*100)<<"%) "<<bx.cl
                     <<": "<<bx.x<<" "<<bx.y<<" "<<bx.w<<" "<<bx.h<<"\n";

            cv::rectangle(img, cv::Point(bx.x - bx.w/2, bx.y - bx.h/2), 
                               cv::Point(bx.x + bx.w/2, bx.y + bx.h/2),
            cv::Scalar( 0, 0, 255), 2);
        }

        //show results
        if(show) {
            cv::namedWindow("result");
            cv::imshow("result", img);
            cv::waitKey(1);
        }

	processed_images++;   
	std::cout<<"mean time per frames: "<<mTime/processed_images/1000<<" ms\n"<<"\n"; 
    }

    return 0;
}
