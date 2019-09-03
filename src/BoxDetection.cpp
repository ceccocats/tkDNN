#include "BoxDetection.h"
#include <string.h>
char buf_frame_crop_name [200];

cv::Mat img_threshold(cv::Mat frame_crop)
{
    // Image Threshold Example
    // https://docs.opencv.org/3.4/d7/d1c/tutorial_js_watershed.html
    cv::Mat f = frame_crop.clone();
    cv::Mat dst, gray;
    // gray and threshold image
    cv::cvtColor(f, gray, cv::COLOR_RGBA2GRAY, 0);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    return gray;
}

cv::Mat img_background(cv::Mat frame_crop)
{
    // Image Background Example
    // https://docs.opencv.org/3.4/d7/d1c/tutorial_js_watershed.html
    cv::Mat f = frame_crop.clone();
    cv::Mat dst, gray, opening, coinsBg;
    // gray and threshold image
    cv::cvtColor(f, gray, cv::COLOR_RGBA2GRAY, 0);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    // get background
    cv::Mat M = cv::Mat(3, 3, CV_8U, cv::Scalar(1,1,1,1));
    cv::erode(gray, opening, M);
    cv::dilate(gray, opening, M);
    cv::Point p = cv::Point(-1,-1);
    cv::dilate(opening, coinsBg, M, p, 3);
    return coinsBg;
}

cv::Mat img_dist_transform(cv::Mat frame_crop)
{
    // Distance Transform Example
    // https://docs.opencv.org/3.4/d7/d1c/tutorial_js_watershed.html
    cv::Mat f = frame_crop.clone();
    cv::Mat dst, gray, opening, coinsBg, coinsFg, distTrans;
    // gray and threshold image
    cv::cvtColor(f, gray, cv::COLOR_RGBA2GRAY, 0);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    // cv::Mat::ones M(3,3,cv::CV_8U);
    // get background
    cv::Mat M = cv::Mat(3, 3, CV_8U, cv::Scalar(1,1,1,1));
    cv::erode(gray, opening, M);
    cv::dilate(gray, opening, M);
    cv::Point p = cv::Point(-1,-1);
    cv::dilate(opening, coinsBg, M, p, 3);
    // distance transorm
    cv::distanceTransform(opening, distTrans, cv::DIST_L2, 5);
    cv::normalize(distTrans, distTrans, 1, 0, cv::NORM_INF);
    return distTrans;
}

// cv::Mat img_watershed(cv::Mat frame_crop)
// {
//     // Image Watershed Example
//     // https://docs.opencv.org/3.4/d7/d1c/tutorial_js_watershed.html
//     cv::Mat f = frame_crop.clone();
//     cv::Mat dst, gray, opening, coinsBg, coinsFg, distTrans, unknown, markers;
//     // gray and threshold image
//     cv::cvtColor(f, gray, cv::COLOR_RGBA2GRAY, 0);
//     cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
//     // get background
//     cv::Mat M = cv::Mat(3, 3, CV_8U, cv::Scalar(1,1,1,1));
//     cv::erode(gray, opening, M);
//     cv::dilate(gray, opening, M);
//     cv::Point p = cv::Point(-1,-1);
//     cv::dilate(opening, coinsBg, M, p, 3);
//     // distance transorm
//     cv::distanceTransform(opening, distTrans, cv::DIST_L2, 5);
//     cv::normalize(distTrans, distTrans, 1, 0, cv::NORM_INF);

//     // get foreground
//     cv::threshold(distTrans, coinsFg, 0.7 * 1, 255, cv::THRESH_BINARY);
//     coinsFg.convertTo(coinsFg, CV_8U, 1, 0); 
//     cv::subtract(coinsBg, coinsFg, unknown);
//     // get connected components networks
//     cv::connectedComponents(coinsFg, markers);
//     // intptr_t n = NULL; 
//     for(int i = 0; i< markers.rows; i++)
//     {
//         for (int j = 0; j< markers.cols; j++)
//         {
//             M.at<uchar>(0, 0);
//             markers.intPtr(i,j)[0] = markers.ucharPtr(i,j)[0] +1;
//             if(unknown.ucharPtr(i,j)[0] == 255)
//             {
//                 markers.intPtr(i,j)[0] = 0;
//             }
//         }
//     }
//     cv::cvtColor(f, f, cv::COLOR_RGBA2RGB, 0);
//     cv::watershed(f, markers);
//     //draw barriers
//     for(int i = 0; i< markers.rows; i++)
//     {
//         for (int j = 0; j< markers.cols; j++)
//         {
//             if(markers.IntPtr(i,j)[0] == -1)
//             {
//                 f.ucharPtr(i,j)[0] = 255; // R
//                 f.ucharPtr(i,j)[1] = 0; // G
//                 f.ucharPtr(i,j)[2] = 0; // B
//             }
//         }
//     }
// }

//////

cv::Mat img_sobel_abssobel(cv::Mat frame_crop, int ret=0)
{
    //ret = 0 --> dstx
    //ret = 1 --> dsty
    //ret = 2 --> absDstx
    //ret = 3 --> absDsty
    // Image Sobel and Image AbsSobel
    // https://docs.opencv.org/trunk/da/d85/tutorial_js_gradients.html
    // compute image gradient on two different directions
    
    cv::Mat f = frame_crop.clone();
    int x,y;
    (ret == 0 || ret == 2)?x=1, y=0 : NULL; 
    (ret == 1 || ret == 3)?x=0, y=1 : NULL; 
    cv::Mat dst; 
    cv::cvtColor(f, f, cv::COLOR_RGB2GRAY, 0);
    // You can try more different parameters
    cv::Sobel(f, dst, CV_8U, x, y, 3, 1, 0, cv::BORDER_DEFAULT);
    // for absSobel
    if(ret == 2 || ret == 3)
        cv::convertScaleAbs(dst, dst, 1, 0);
    // next 3 rows to be checked
    //// ??cv::Mat f2 = frame_crop.clone();
    //// cv.Scharr(?(f,f2), dstx, cv.CV_8U, 1, 0, 1, 0, cv.BORDER_DEFAULT);
    //// cv.Scharr(?(f,f2), dsty, cv.CV_8U, 0, 1, 1, 0, cv.BORDER_DEFAULT);    
    return dst;
}

cv::Mat img_laplacian(cv::Mat frame_crop, int ret=1)
{
    //ret = 0 --> src_gray
    //ret = 1 --> dst
    // Image Laplacian
    // compute image gradient with laplacian 
    cv::Mat f = frame_crop.clone();
    cv::Mat src_gray, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    cv::GaussianBlur( f, f, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    /// Convert the image to grayscale
    cv::cvtColor( f, src_gray, CV_RGB2GRAY );
    if (ret == 0)
        return src_gray;

    // else: Apply Laplace function
    cv::Mat abs_dst;
    cv::Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT );
    // //compute sharpness
    // float sharpnessValue = cv::mean(dst);
    return dst;    
}

cv::Mat find_contours(cv::Mat frame_crop, cv::Mat img, cv::Mat canny_output, int n_lines=1)
{
    // n_line: number of line to plot on image
    cv::Mat img_line = frame_crop.clone();
    cv::Mat ret_thresh;
    std::vector<std::vector<cv::Point> > contours;
    double thresh = 127;
    double maxValue = 255;
    cv::threshold(img, ret_thresh, thresh, maxValue, 0);//0); // = cv2.threshold(img,127,255,0)
    cv::findContours(canny_output, contours, 1, 2);//cv::CHAIN_APPROX_SIMPLE );//1, 2); //contours,hierarchy = cv2.findContours(thresh, 1, 2)
    // cv::threshold(img2, ret2, thresh, maxValue, 0);
    // cv::findContours(canny_output2, contours2, 1, 2);
    // cv::threshold(img3a, ret3a, thresh, maxValue, 0);
    // cv::findContours(canny_output3a, contours3a, 1, 2);
    // cv::threshold(img3b, ret3b, thresh, maxValue, 0);
    // cv::findContours(canny_output3b, contours3b, 1, 2);

    cv::Vec4f line;
    float vx,vy,x,y;
    int lefty, righty;
    for(int i=0; i<n_lines; i++)
    {
        cv::fitLine(contours[i],line,CV_DIST_L2,0,0.01,0.01);
        vx = line(0);
        vy = line(1);
        x = line(2);
        y = line(3); 
        lefty = int((-x*vy/vx) + y);
        righty = int(((img.cols-x)*vy/vx)+y);
        cv::line(img_line,cv::Point(img.cols-1,righty),cv::Point(0,lefty),(255, 0 ,0),2);
    }
    
    // cv::imshow("bla", img);
    // cv::waitKey(1000);
    return img_line;    
}

// cv::Mat fit_rectangular(cv::Mat frame_crop, cv::Mat img, cv::Mat canny_output)
// {
//     cv::Mat img_clone = frame_crop.clone();
//     cv::Mat ret_thresh;
//     std::vector<std::vector<cv::Point> > contours;
//     double thresh = 127;
//     double maxValue = 255;
//     cv::threshold(img, ret_thresh, thresh, maxValue, 0);//0); // = cv2.threshold(img,127,255,0)
//     cv::findContours(canny_output, contours, 1, 2);//cv::CHAIN_APPROX_SIMPLE );//1, 2); //contours,hierarchy = cv2.findContours(thresh, 1, 2)

//     cv::RotatedRect rect = cv::minAreaRect(contours[0]);
//     cv::Mat boxPts1;
//     std::vector<std::vector<cv::Point> > boxPts2;
//     cv::boxPoints(rect, boxPts1); 
//     // boxPts = np.int0(boxPts);
//     for (int x = 0; x < img.cols; x++)
//         for (int y = 0; y < img.rows; y++)
//             boxPts2.at(x).push_back(cv::Point(boxPts1.at<int>(x, y)));

//     cv::drawContours(img_clone, boxPts2,0,(0,0,255),2);
//     // drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
//     return img_clone;
// }

cv::Mat compute_saliency(cv::Mat frame_crop, cv::Ptr<cv::saliency::Saliency>  saliencyAlgorithm, int const_molt_mat, int ret=0)
{
    //ret=0 --> saliencyMap
    //ret=1 --> binaryMap
    // SPECTRAL_RESIDUAL algorithm
    cv::Mat f = frame_crop.clone();
    cv::Mat saliencyMap;
    cv::Mat binaryMap;
    
    if( saliencyAlgorithm->computeSaliency( f, saliencyMap ) )
    {
        if(ret==0)
            return saliencyMap*const_molt_mat;

        cv::saliency::StaticSaliencySpectralResidual spec;
        spec.computeBinaryMap( saliencyMap, binaryMap );

        // imshow( "Saliency Map", saliencyMap );
        // imshow( "Original Image", image );
        // imshow( "Binary Map", binaryMap );
        // waitKey( 0 );
        return binaryMap*const_molt_mat;
    }
    return cv::Mat(0,0,CV_8U, cv::Scalar(0,0,0,0));
}

//////

void image_segmentation(cv::Mat frame_crop, int frame_nbr, int i)
{    
    // Watershed Algorithm
    // https://docs.opencv.org/3.4/d7/d1c/tutorial_js_watershed.html
    
    auto step_t_segmentation = std::chrono::steady_clock::now();
    auto end_t_segmentation = std::chrono::steady_clock::now();
    cv::Mat ret;
    // ret = img_threshold(frame_crop);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imgthr.jpg", frame_nbr, i, img_threshold(frame_crop));
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME imgthr ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // ret =img_background(frame_crop);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imgback.jpg", frame_nbr, i, img_background(frame_crop));
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME imgback ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // ret = img_dist_transform(frame_crop);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imgtrans.jpg", frame_nbr, i, img_dist_transform(frame_crop));
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME imgtrans ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // // ret = img_watershed(frame_crop);
    // if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imgwatershed.jpg", frame_nbr, i, img_watershed(frame_crop)); 
}

void image_gradients(cv::Mat frame_crop, int frame_nbr, int i)
{
    // Image Gradients
    // https://docs.opencv.org/trunk/da/d85/tutorial_js_gradients.html
    
    auto step_t_segmentation = std::chrono::steady_clock::now();
    auto end_t_segmentation = std::chrono::steady_clock::now();
    cv::Mat ret;
    // sobel
    // ret = img_sobel_abssobel(frame_crop, 0);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imgsobel_x_8U.jpgg", frame_nbr, i, img_sobel_abssobel(frame_crop, 0));

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME sobel0 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // ret = img_sobel_abssobel(frame_crop, 1);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imgsobel_y_8U.jpg", frame_nbr, i, img_sobel_abssobel(frame_crop, 1));

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME sobel1 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // ret = img_sobel_abssobel(frame_crop, 2);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imgsobel_x_64F.jpg", frame_nbr, i, img_sobel_abssobel(frame_crop, 2));

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME sobel2 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // ret = img_sobel_abssobel(frame_crop, 3);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imgsobel_y_64F.jpg", frame_nbr, i, img_sobel_abssobel(frame_crop, 3));

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME sobel3 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    
    // laplacian
    // ret = img_laplacian(frame_crop, 0);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imglaplacian_gr.jpg", frame_nbr, i, img_laplacian(frame_crop, 0));

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME laplacian0 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation; 
    // ret = img_laplacian(frame_crop, 1);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_imglaplacian_dst.jpg", frame_nbr, i, img_laplacian(frame_crop, 1));

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME laplacian1 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    //////
    
}    

void image_find_contours(cv::Mat frame_crop, int frame_nbr, int i)
{
    // Finding contours in your image
    // https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html

    
    auto step_t_segmentation = std::chrono::steady_clock::now();
    auto end_t_segmentation = std::chrono::steady_clock::now();
    // plot lines on figure. 3 ways:
    // 1 - use gray image (no more operations) to get contours (one line)
    // 2 - use laplacian image (one line)
    // 3 - use sobel (1st dir) image and sobel (2nd dir) image to plot two different lines
    cv::Mat canny_output1, canny_output2, canny_output3a, canny_output3b;
    cv::Mat contours;
    // src_gray
    cv::Mat img1 = img_laplacian(frame_crop, 0);
    cv::Canny(img1, canny_output1, 100, 100*2 );
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_canny1.jpg", frame_nbr, i, canny_output1);

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME canny1 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // // dst
    // cv::Mat img2 = img_laplacian(frame_crop, 2);
    // cv::Canny(img2, canny_output2, 100, 100*2 );
    cv::Canny(img1, canny_output2, 100, 100*2 );
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_canny2.jpg", frame_nbr, i, canny_output2);

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME canny2 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // dstx
    cv::Mat img3a = img_sobel_abssobel(frame_crop, 0);
    cv::Canny(img3a, canny_output3a, 100, 100*2 );
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_canny3a.jpg", frame_nbr, i, canny_output3a);

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME canny3a ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // dsty
    cv::Mat img3b = img_sobel_abssobel(frame_crop, 1);
    cv::Canny(img3b, canny_output3b, 100, 100*2 );
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_canny3b.jpg", frame_nbr, i, canny_output3b);

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME canny3b ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    
    // 1 line
    // contours = find_contours(frame_crop, img1, canny_output1, 1);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_line1.jpg", frame_nbr, i, find_contours(frame_crop, img1, canny_output1, 1));

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME line1 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // 3 line
    // contours = find_contours(frame_crop, img1, canny_output2, 1);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_line2.jpg", frame_nbr, i, find_contours(frame_crop, img1, canny_output2, 1));

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME line2 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    // mix 1 line of image with 1 line of another
    cv::Mat img_line = frame_crop.clone();
    img_line = find_contours(img_line, img3a, canny_output3a, 1);
    img_line = find_contours(img_line, img3b, canny_output3b, 1);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_line3.jpg", frame_nbr, i, img_line);

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME line3 ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;

    // img_line = frame_crop.clone();
    // img_line = find_contours(img_line, img3a, canny_output3a, 2);
    // img_line = find_contours(img_line, img3b, canny_output3b, 2);
    // sprintf(buf_frame_crop_name,"../demo/demo/data/img_crop/%d_%d_line3bis.jpg",frame_nbr, i);
    // cv::imwrite(buf_frame_crop_name, img_line);
    
    
    // cv::Mat canny_output4;
    // cv::Mat img4 = img_laplacian(frame_crop, 0);
    // cv::Canny(img4, canny_output4, 100, 100*2 );
    // printf(buf_frame_crop_name,"../demo/demo/data/img_crop/%d_%d_rect.jpg",frame_nbr, i);
    // cv::imwrite(buf_frame_crop_name, fit_rectangular(frame_crop, img4, canny_output4));
}


void image_saliency(cv::Mat frame_crop, int frame_nbr, int i)
{
    // https://github.com/opencv/opencv_contrib/blob/master/modules/saliency/samples/computeSaliency.cpp
    cv::Ptr<cv::saliency::Saliency> saliencyAlgorithm;
    
    int const_molt_mat = 0;
    auto step_t_segmentation = std::chrono::steady_clock::now();
    auto end_t_segmentation = std::chrono::steady_clock::now();

    // SPECTRAL_RESIDUAL
    const_molt_mat = 255;
    saliencyAlgorithm = cv::saliency::StaticSaliencySpectralResidual::create();
    cv::Mat spect_res = compute_saliency(frame_crop, saliencyAlgorithm, const_molt_mat, 0);
    if(!spect_res.empty())
    {
        if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_saliency_SpectralResidual.jpg", frame_nbr, i, spect_res);
    }
    else
    {
        std::cout<<"something is wrond (image_saliency)"<<std::endl;
    }
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME SPECTRAL_RESIDUAL ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;

    // BINARY SPECTRAL_RESIDUAL
    const_molt_mat = 255;
    spect_res = compute_saliency(frame_crop, saliencyAlgorithm, const_molt_mat, 1);
    if(!spect_res.empty())
    {
        if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_saliency_BinarySpectralResidual.jpg", frame_nbr, i, spect_res);
    }
    else
    {
        std::cout<<"something is wrond (image_saliency)"<<std::endl;
    }
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME BINARY SPECTRAL_RESIDUAL ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;

    // FINE_GRAINED
    const_molt_mat = 1;
    saliencyAlgorithm =  cv::saliency::StaticSaliencyFineGrained::create();
    spect_res = compute_saliency(frame_crop, saliencyAlgorithm, const_molt_mat, 0);
    if(!spect_res.empty())
    {
        if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_saliency_FineGrained.jpg", frame_nbr, i, spect_res);
    }
    else
    {
        std::cout<<"something is wrond (image_saliency)"<<std::endl;
    }
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME FINE_GRAINED ("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;

    // saliencyAlgorithm = cv::saliency::ObjectnessBING::create();
    // std::vector<cv::Vec4i> saliencyMap1;
    // saliencyAlgorithm.dynamicCast<cv::saliency::ObjectnessBING>()->setTrainingPath( "" );
    // saliencyAlgorithm.dynamicCast<cv::saliency::ObjectnessBING>()->setBBResDir( "Results" );
    // std::cout<<"mmm"<<std::endl;
    // saliencyAlgorithm->computeSaliency( frame_crop, saliencyMap1 );
    // int ndet = int(saliencyMap1.size());
    // std::cout << "Objectness done " << ndet << std::endl;
    // // // The result are sorted by objectness. We only use the first maxd boxes here.
    // // int maxd = 7, step = 255 / maxd, jitter=9; // jitter to seperate single rects
    // // cv::Mat draw = frame_crop.clone();
    // // for (int i = 0; i < std::min(maxd, ndet); i++) 
    // // {
    // //     cv::Vec4i bb = saliencyMap1[i];
    // //     cv::Scalar col = cv::Scalar(((i*step)%255), 100, 255-((i*step)%255));
    // //     cv::Point off(cv::theRNG().uniform(-jitter,jitter), cv::theRNG().uniform(-jitter,jitter));
    // //     cv::rectangle(draw, cv::Point(bb[0]+off.x, bb[1]+off.y), cv::Point(bb[2]+off.x, bb[3]+off.y), col, 2);
    // //     cv::rectangle(draw, cv::Rect(20, 20+i*10, 10,10), col, -1); // mini temperature scale
    // // }
    // // imshow("BING", draw);
    // // waitKey();
    // printf(buf_frame_crop_name,"../demo/demo/data/img_crop/%d_%d_saliency_BING.jpg",frame_nbr, i);
    // cv::imwrite(buf_frame_crop_name, saliencyMap1);

    //// 

    // BING WANG APR 2014
    cv::Mat saliencyMap;
    cv::Mat frame_sal = frame_crop.clone();
    saliencyAlgorithm = cv::saliency::MotionSaliencyBinWangApr2014::create();
    saliencyAlgorithm.dynamicCast<cv::saliency::MotionSaliencyBinWangApr2014>()->setImagesize( frame_sal.cols, frame_sal.rows );
    saliencyAlgorithm.dynamicCast<cv::saliency::MotionSaliencyBinWangApr2014>()->init();
    cvtColor( frame_sal, frame_sal, cv::COLOR_BGR2GRAY );
    saliencyAlgorithm->computeSaliency( frame_sal, saliencyMap);
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d_saliency_BinWangApr.jpg", frame_nbr, i, saliencyMap);

    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " - TIME BING WANG APR 2014("<<frame_nbr<<"-"<<i<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
}

cv::Mat frame_disparity(cv::Mat pre_frame, cv::Mat frame, int frame_nbr, int i, int ret=0)
{
    // https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
    cv::Mat backgroundImage = pre_frame.clone();
    cv::Mat currentImage = frame.clone();
    cv::Mat diffImage;
    // pass to HSV color
    if(ret)
    {
        cv::cvtColor(backgroundImage, backgroundImage, CV_BGR2HSV); 
        cv::cvtColor(currentImage, currentImage, CV_BGR2HSV); 
    }
    cv::absdiff(backgroundImage, currentImage, diffImage);

    cv::Mat foregroundMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);
    // std::cout<<"diffImage: "<<diffImage.cols<<" - "<<diffImage.rows<<std::endl;
    // if(SAVE) SAVE_TO("../demo/demo/data/img_disparity/%d_%d_pc1.jpg", frame_nbr, i, backgroundImage);
    // if(SAVE) SAVE_TO("../demo/demo/data/img_disparity/%d_%d_c1.jpg", frame_nbr, i, currentImage);
    float threshold = 30.0f;
    float dist;

    for(int j=0; j<diffImage.rows; ++j)
    {
        for(int k=0; k<diffImage.cols; ++k)
        {
            cv::Vec3b pix = diffImage.at<cv::Vec3b>(j,k);

            dist = (pix[0]*pix[0] + pix[1]*pix[1] + pix[2]*pix[2]);
            dist = sqrt(dist);

            if(dist>threshold)
            {
                foregroundMask.at<unsigned char>(j,k) = 255;
            }
        }
    }
    if(SAVE) SAVE_TO("../demo/demo/data/img_disparity/%d_%d_dif.jpg", frame_nbr, i, foregroundMask);

    return foregroundMask;
}

void frame_box_disparity(cv::Mat pre_frame, cv::Mat frame, std::vector <cv::Rect>  pre_rois, int frame_nbr)
{

    int roi_tollerance = 10;
    cv::Mat pre_frame_crop, frame_crop;
    int dx, dy;
    int id = 1;
    auto step_t_segmentation = std::chrono::steady_clock::now();
    auto end_t_segmentation = std::chrono::steady_clock::now();

    for(auto r : pre_rois)
    {
        if(SAVE) SAVE_TO("../demo/demo/data/img_disparity/%d_%d_orig.jpg", frame_nbr, id, pre_frame(r));

        //resize last roi with a tollerance
        dx = r.width / roi_tollerance;
        dy = r.height / roi_tollerance;
        r.x = (r.x - dx > 0)? (r.x - dx) : 0;
        r.y = (r.y - dy > 0)? (r.y - dy) : 0;
        // std::cout<<"disp: x "<<r.x<<" - y "<<r.y<<std::endl;
        r.width = ((r.x+r.width+dx+dx) >= frame.cols)? (frame.cols-1-r.x) : (r.width+dx+dx);
        r.height = ((r.y+r.height+dy+dy) >= frame.rows)? (frame.rows-1-r.y) : (r.height+dy+dy);
        // std::cout<<"disp: w "<<r.width<<" - h "<<r.height<<std::endl;
        // std::cout<<"disp: wf "<<frame.cols<<" - hf "<<frame.rows<<std::endl;
        // std::cout<<"---"<<std::endl;
        // std::cout<<"disp: x "<<r.x<<" to "<<r.width+r.x<<" wf "<<frame.cols<<std::endl;
        // std::cout<<"disp: y "<<r.y<<" to "<<r.height+r.y<<" hf "<<frame.rows<<std::endl;
                       
        //crop pre_frame and current frame 
        pre_frame_crop = pre_frame(r);
        frame_crop = frame(r);
        if(SAVE) SAVE_TO("../demo/demo/data/img_disparity/%d_%d_cur.jpg", frame_nbr, id, frame_crop);
        if(SAVE) SAVE_TO("../demo/demo/data/img_disparity/%d_%d_pre.jpg", frame_nbr, id, pre_frame_crop);
        
        // difference from two consecutive frame
        step_t_segmentation = std::chrono::steady_clock::now();
        frame_disparity(pre_frame_crop, frame_crop, frame_nbr, id, 0);
        end_t_segmentation = std::chrono::steady_clock::now();
        std::cout << " TIME frame_disparity ("<<frame_nbr<<"-"<<id<<") : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
        step_t_segmentation = end_t_segmentation;
        id++;
    }
}

void segmentation(cv::Mat pre_frame, cv::Mat frame_crop, int frame_nbr, int i, int mode)
{
    //mode=0 (for whole frame), it computes the frame disparity
    //mode=1 (for single box), it doesn't compute the frame disparity (it has already been done-see frame_box_disparity())
    // whole figure
    char buf_str [15];
    if(!mode)
        sprintf(buf_str,"whole frame");
    else
        sprintf(buf_str,"a box frame");
    
    if(SAVE) SAVE_TO("../demo/demo/data/img_crop/%d_%d.jpg", frame_nbr, i, frame_crop);

    auto step_t_segmentation = std::chrono::steady_clock::now();
    auto end_t_segmentation = std::chrono::steady_clock::now();

    // Watershed Algorithm
    std::cout<<"image segmentation:"<<std::endl;
    image_segmentation(frame_crop, frame_nbr, i);
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " TIME "<<buf_str<<": image_segmentation : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;

    // Image Gradients
    std::cout<<"image gradients:"<<std::endl;
    image_gradients(frame_crop, frame_nbr, i);
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " TIME "<<buf_str<<": image_gradients : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;

    // Find contours
    std::cout<<"image find contours:"<<std::endl;
    image_find_contours(frame_crop, frame_nbr, i);
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " TIME "<<buf_str<<": image_find_contours : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;
    
    //saliency map
    std::cout<<"image saliency:"<<std::endl;
    image_saliency(frame_crop, frame_nbr, i);
    end_t_segmentation = std::chrono::steady_clock::now();
    std::cout << " TIME "<<buf_str<<": image_saliency : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
    step_t_segmentation = end_t_segmentation;

    //frame disparity
    if(!mode && frame_nbr!=0)
    {
        std::cout<<"frame disparity:"<<std::endl;
        frame_disparity(pre_frame, frame_crop, frame_nbr, i, 0);
        end_t_segmentation = std::chrono::steady_clock::now();
        std::cout << " TIME "<<buf_str<<": frame_disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
        step_t_segmentation = end_t_segmentation;
    }
    

}