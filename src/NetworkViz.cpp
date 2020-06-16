#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tkDNN/NetworkViz.h"

namespace tk { namespace dnn {

cv::Mat vizFloat2colorMap(cv::Mat map) {

    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    // expand your range to 0..255. Similar to histEq();
    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min); 
    //return adjMap;

    
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
    return falseColorsMap;
}

cv::Mat vizData2Mat(dnnType *dataInput, tk::dnn::dataDim_t dim, int imgdim) {
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
        cv::Mat raw = vizFloat2colorMap(cv::Mat(cv::Size(dim.w, dim.h),CV_32FC1, data + dim.w*dim.h*i));
        int r = i / gridDim;
        int c = i - r * gridDim;
        raw.copyTo(grid.rowRange(r*dim.h, r*dim.h + dim.h).colRange(c*dim.w, c*dim.w + dim.w));
    }

    float ar = float(dim.w)/dim.h;
    cv::Size vdim(ar*imgdim, imgdim);
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
    return vizData2Mat(net->layers[layer]->dstData, net->layers[layer]->output_dim, imgdim);

    //cv::imwrite("viz/layer" + std::to_string(layer) + ".png", viz);
    //cv::imshow("layer", viz);
    //cv::waitKey(0);
}

}}