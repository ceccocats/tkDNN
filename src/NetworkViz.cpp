#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tkDNN/NetworkViz.h"

namespace tk { namespace dnn {

cv::Mat mapillary_15_map(cv::Mat adjMap){

    // cv::imshow("test", adjMap);
    // cv::waitKey(0);
    cv::Mat M1(1, 256, CV_8UC1), M2(1, 256, CV_8UC1), M3(1, 256, CV_8UC1);

    M3.at<uchar>(0)=165;
    M2.at<uchar>(0)=42;
    M1.at<uchar>(0)=45;

    M3.at<uchar>(1)=196;
    M2.at<uchar>(1)=196;
    M1.at<uchar>(1)=196;

    M3.at<uchar>(2)=90;
    M2.at<uchar>(2)=120;
    M1.at<uchar>(2)=150;

    M3.at<uchar>(3)=128;
    M2.at<uchar>(3)=64;
    M1.at<uchar>(3)=128;

    M3.at<uchar>(4)=70;
    M2.at<uchar>(4)=70;
    M1.at<uchar>(4)=70;

    M3.at<uchar>(5)=220;
    M2.at<uchar>(5)=20;
    M1.at<uchar>(5)=60;

    M3.at<uchar>(6)=255;
    M2.at<uchar>(6)=255;
    M1.at<uchar>(6)=255;

    M3.at<uchar>(7)=107;
    M2.at<uchar>(7)=142;
    M1.at<uchar>(7)=35;

    M3.at<uchar>(8)=70;
    M2.at<uchar>(8)=130;
    M1.at<uchar>(8)=180;

    M3.at<uchar>(9)=220;
    M2.at<uchar>(9)=220;
    M1.at<uchar>(9)=220;

    M3.at<uchar>(10)=153;
    M2.at<uchar>(10)=153;
    M1.at<uchar>(10)=153;

    M3.at<uchar>(11)=128;
    M2.at<uchar>(11)=128;
    M1.at<uchar>(11)=128;

    M3.at<uchar>(12)=119;
    M2.at<uchar>(12)=11;
    M1.at<uchar>(12)=32;

    M3.at<uchar>(13)=0;
    M2.at<uchar>(13)=0;
    M1.at<uchar>(13)=142;

    for(int i=14;i<256;i++)
    {
        M1.at<uchar>(i)=0;
        M2.at<uchar>(i)=0;
        M3.at<uchar>(i)=0;
    }

    cv::Mat r1,r2,r3;

    cv::LUT(adjMap,M1,r1);
    cv::LUT(adjMap,M2,r2);
    cv::LUT(adjMap,M3,r3);

    std::vector<cv::Mat> planes;
    planes.push_back(r1);
    planes.push_back(r2);
    planes.push_back(r3);

    cv::Mat dst;
    cv::merge(planes,dst);
    return dst;


}

cv::Mat vizFloat2colorMap(cv::Mat map,double min, double max, bool mapillary_15) {

    if(min == 0 && max == 0)
        cv::minMaxIdx(map, &min, &max);

    cv::Mat adjMap;
    cv::Mat falseColorsMap;
    
    if(mapillary_15){
        map.convertTo(adjMap,CV_8UC1);
        falseColorsMap = mapillary_15_map(adjMap);
    }
    else{
        // expand your range to 0..255. Similar to histEq();
        map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min); 
        applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);
    }
    return falseColorsMap;
}

cv::Mat vizData2Mat(dnnType *dataInput, tk::dnn::dataDim_t dim, int img_h, int img_w, double min, double max, bool mapillary_15) {
    dnnType *data = nullptr;

    // copy to CPU
    if(isCudaPointer(dataInput)) {
        data = new dnnType[dim.tot()];
        checkCuda( cudaMemcpy(data, dataInput, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToHost) );
    } else {
        data = dataInput;
    }

    int gridDim = ceil(sqrt(dim.c));
    cv::Size gridSize(dim.w*gridDim, dim.h*gridDim);
    cv::Mat grid = cv::Mat(gridSize, CV_8UC3, cv::Scalar(0));    

    for(int i=0; i<dim.c;i++) {
        cv::Mat raw = vizFloat2colorMap(cv::Mat(cv::Size(dim.w, dim.h),CV_32FC1, data + dim.w*dim.h*i), min, max, mapillary_15);
        int r = i / gridDim;
        int c = i - r * gridDim;
        raw.copyTo(grid.rowRange(r*dim.h, r*dim.h + dim.h).colRange(c*dim.w, c*dim.w + dim.w));
    }

    cv::Size vdim(img_w, img_h);
    cv::Mat viz;
    cv::resize(grid, viz, vdim, 0, 0, 0);
    
    // free memory
    if(isCudaPointer(dataInput)) {
        delete [] data;
    }
    return viz;
}

cv::Mat vizLayer2Mat(tk::dnn::Network *net, int layer, int imgdim) {
    if(layer >= net->num_layers)
        FatalError("Could not viz layer\n");
    return vizData2Mat(net->layers[layer]->dstData, net->layers[layer]->output_dim, imgdim, imgdim);

    //cv::imwrite("viz/layer" + std::to_string(layer) + ".png", viz);
    //cv::imshow("layer", viz);
    //cv::waitKey(0);
}

}}