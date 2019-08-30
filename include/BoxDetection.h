#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <time.h>
#include <chrono>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include <cublas_v2.h>
#include <cudnn.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//saliency
#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>

// cv::Mat img_threshold(cv::Mat frame_crop);
// cv::Mat img_background(cv::Mat frame_crop);
// cv::Mat img_dist_transform(cv::Mat frame_crop);
// cv::Mat img_watershed(cv::Mat frame_crop);
void image_segmentation(cv::Mat frame_crop, int frame_nbr, int i);
void image_gradients(cv::Mat frame_crop, int frame_nbr, int i);
void image_find_contours(cv::Mat frame_crop, int frame_nbr, int i);
void image_saliency(cv::Mat frame_crop, int frame_nbr, int i);
void frame_box_disparity(cv::Mat pre_frame, cv::Mat frame, std::vector <cv::Rect>  pre_rois, int frame_nbr);
void segmentation(cv::Mat pre_frame, cv::Mat frame_crop, int frame_nbr, int i, int mode);

//canny
cv::Mat img_laplacian(cv::Mat frame_crop, int ret);
void frame_disparity(cv::Mat pre_frame, cv::Mat frame, int frame_nbr, int i, int ret);