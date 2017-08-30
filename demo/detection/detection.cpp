#include<iostream>
#include "tkdnn.h"
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const char *reg_bias = "../tests/yolo/layers/g31.bin";

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

void compute_image( cv::Mat imageORIG, 
                    tkDNN::NetworkRT *netRT, tkDNN::RegionInterpret *rI,
                    dnnType *input, dnnType *output) {

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
    printCenteredTitle(" TENSORRT inference ", '=', 30); 
    TIMER_START
    checkCuda(  cudaMemcpyAsync(netRT->buffersRT[netRT->buf_input_idx], input, 
                                netRT->input_dim.tot()*sizeof(float), 
                                cudaMemcpyHostToDevice, netRT->stream));
    netRT->enqueue();
    checkCuda(  cudaMemcpyAsync(output, netRT->buffersRT[netRT->buf_output_idx], 
                                netRT->output_dim.tot()*sizeof(float), 
                                cudaMemcpyDeviceToHost, netRT->stream));
    cudaStreamSynchronize(netRT->stream);
    TIMER_STOP
    

    rI->interpretData(output, imageORIG.cols, imageORIG.rows);

}

int print_usage() {
    std::cout<<"usage: ./detection net.rt validation_list.txt [-t <thresh>] [-s]\n"
             <<" -t: set thresh value\n -s: show images as compute\n\n" 
             <<"> validation_list.txt format: \n"
             <<"    path/to/image.jpg path/to/label.txt\n"
             <<"> label.txt format: \n"  
             <<"    <object-class> <x> <y> <width> <height>\n"
             <<"    x and y are the box center, "   
             <<"all values are relative to the image size\n\n";
    return 1;
}


int main(int argc, char *argv[]) {

    //params
    char *tensor_path = NULL;
    char *imageset_path = NULL; 
    float thresh = 0.3f;
    bool  show = false;

    //parse params
    int c;
    while ((c = getopt (argc, argv, "t:s")) != -1) {
        switch(c) {
        case 't':   thresh = atof(optarg); break;
        case 's':   show = true;           break;
        case '?':   
            return print_usage();
        default:    return print_usage();
        }
    }

    if(argc - optind == 2) {
        tensor_path   = argv[optind];
        imageset_path = argv[optind+1];
    } else {
        std::cout<<"not enough arguments.\n";
        return print_usage();
    }
    //end parsing

    if(!fileExist(tensor_path))
        FatalError("unable to read serialRT file");
    //convert network to tensorRT
    tkDNN::NetworkRT netRT(NULL, tensor_path);
    tkDNN::RegionInterpret rI(netRT.input_dim, netRT.output_dim, 80, 4, 5, thresh, reg_bias);

    dnnType *input = new float[netRT.input_dim.tot()];
    dnnType *output = new float[netRT.output_dim.tot()];

    std::string line;
    std::ifstream imageset(imageset_path);
    if(!imageset.is_open())
        FatalError("could not read imageset");
    
    float mAP = 0;
    int processed_images;
    for(processed_images=1; getline(imageset, line); processed_images++) {
        
        std::string image_path = line.substr(0, line.find(" "));
        std::string label_path = line.substr(line.find(" ")+1, line.size());
        std::cout<<image_path<<"\n"<<label_path<<"\n";

        //LOAD IMAGE
        cv::Mat img = cv::imread(image_path.c_str(), CV_LOAD_IMAGE_COLOR);
        if(!img.data)                              
            FatalError("Could not open image");
        std::cout<<"Image size: ("<<img.cols<<"x"<<img.rows<<")\n";

        compute_image(img, &netRT, &rI, input, output); 

        std::ifstream labels(label_path.c_str());
        if(!labels.is_open())
            FatalError("could not read labels");


        qsort(rI.res_boxes, rI.res_boxes_n, sizeof(tkDNN::box), prob_sort);
        for(int i=0; i<rI.res_boxes_n; i++) {
            tkDNN::box bx = rI.res_boxes[i];
            std::cout<<" ("<<int(bx.prob*100)<<"%) "<<bx.cl
                     <<": "<<bx.x<<" "<<bx.y<<" "<<bx.w<<" "<<bx.h<<"\n";

            cv::rectangle(img, cv::Point(bx.x - bx.w/2, bx.y - bx.h/2), 
                               cv::Point(bx.x + bx.w/2, bx.y + bx.h/2),
            cv::Scalar( 0, 0, 255), 2);
        }

        std::cout<<"GROUND TRUTH\n";
        tkDNN::box gt[256];
        int gt_n = 0; 
        int cl;
        float x, y, w, h;
        while(labels>>cl) {
            labels>>x>>y>>w>>h;
            w *= img.cols; x *= img.cols; 
            h *= img.rows; y *= img.rows; 
            std::cout<<cl<<": "<<x<<" "<<y<<" "<<w<<" "<<h<<"\n";
            gt[gt_n].x = x;
            gt[gt_n].y = y;
            gt[gt_n].w = w;
            gt[gt_n].h = h;
            gt[gt_n].cl = cl;
            gt_n++;            

            cv::rectangle(img, cv::Point(x -w/2, y -h/2), 
                               cv::Point(x +w/2, y +h/2),
            cv::Scalar( 255, 0, 0), 2);
        }

        //AP calculation
        float AP = 0;
        for(int i=rI.res_boxes_n; i>=1; i--) {  //for each detected evaluate sub group
            
            int prec = 0;
            for(int j=0; j<i; j++) {            //for each detected in sub group
                for(int z=0; z<gt_n; z++) {     //control each ground truth
                    float iou = tkDNN::RegionInterpret::box_iou(rI.res_boxes[j], gt[z]);
                    if(iou > 0.6f && rI.res_boxes[j].cl == gt[z].cl) {
                        prec++;
                        break;
                    }
                }
            }

            AP += float(prec)/i;
        }
        AP = AP/gt_n;
        std::cout<<"AP: "<<AP<<"\n";        
        
        mAP += AP;
        std::cout<<"#### processed: "<<processed_images
                 <<", mAP: "<<mAP/processed_images<<"\n";        

        //show results
        if(show) {
            cv::namedWindow("result");
            cv::imshow("result", img);
            cv::waitKey(1000);
        }        
    }

    return 0;
}
