#include <iostream>
#include <signal.h>
#include <stdlib.h> /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <ctime>
#include <pthread.h>

#include <time.h>
#include <chrono>
#include <math.h>
#include <typeinfo>

#include "utils.h"
#include "BoxDetection.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//saliency
#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>

#include "Yolo3Detection.h"
#include "classutils.h"
#include "../masa_protocol/include/send.hpp"
#include "../masa_protocol/include/serialize.hpp"

#include "ekf.h"
#include "trackutils.h"
#include "plot.h"
#include "tracker.h"

#define MAX_DETECT_SIZE 100

bool gRun;
std::string obj_class[10]{"person", "car", "truck", "bus", "motor", "bike", "rider", "traffic light", "traffic sign", "train"};

// sem for mainthread, detectionthread and topviewthread
// sem_vc for mainthread, videocapturethread, originalthread and disparitythread
std::mutex sem, sem_vc;
// a single mutex for each operation - the show_updates function must get all mutex
std::mutex mutex_o, mutex_de, mutex_t, mutex_di;

struct ModFrame_t{
    std::vector<Tracker> trackers;
    geodetic_converter::GeodeticConverter gc;
    double adfGeoTransform[6];
    cv::Mat H;
    cv::Mat original_frame;
    tk::dnn::Yolo3Detection yolo;
    cv::Mat mask;
};

struct Frame_t{
    char *input;
    cv::Mat frame;
    int frame_nbr;
};

struct Show_t{
    cv::Mat original, detection, topview, disparity;
    bool update_o, update_de, update_t, update_di; 
}updates;

void sig_handler(int signo)
{
    std::cout << "request gateway stop\n";
    gRun = false;
}

void *readVideoCapture(void *x_void_ptr)
{
    Frame_t *info_f = (Frame_t *) x_void_ptr;
    cv::VideoCapture cap(info_f->input);
    cv::Mat frame_loc;
    int frame_nbr_loc = 0;
    // bool to_show = false;
    if (!cap.isOpened())
        gRun = false;
    else
        std::cout << "camera started\n";
    

    cap.set(cv::CAP_PROP_BUFFERSIZE,3);
    // cap.set(CV_CAP_PROP_FPS, 10);
    // // CV_CAP_PROP_BUFFERSIZE
    std::cout<<"buf size: "<<cap.get(CV_CAP_PROP_BUFFERSIZE)<<std::endl;

    // ModFrame_t *info_show = (ModFrame_t *) x_void_ptr;
    auto start_t = std::chrono::steady_clock::now();
    auto step_t = std::chrono::steady_clock::now();
    auto end_t = std::chrono::steady_clock::now();

    // compute fps
    // std::cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << cap.get(CV_CAP_PROP_FPS) << std::endl;

    // //compute frame rate 
    // int i=0, num_f = 120;
    // while (i<21)
    // {        
    //     cap >> frame;
    //     i++;
    // }

    // i=0;
    // start_t = std::chrono::steady_clock::now();
    // while (i<num_f)
    // {        
    //     step_t = std::chrono::steady_clock::now();

    //     cap >> frame;
    //     i++;
    //     std::cout << " step "<<i<<" : "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - step_t).count() << " ms"<<std::endl;
        
    // }
    // end_t = std::chrono::steady_clock::now();    

    // std::cout << "Capturing " << num_f << " frames" << std::endl ;
    // std::cout << " Time taken : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << " ms"<<std::endl;
   
    // return 0;
    //------------------------

    while (gRun)
    {        
        cap >> frame_loc;
        // CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
        std::cout<<"id: "<<cap.get(cv::CAP_PROP_POS_MSEC)<<std::endl;
        // CAP_PROP_FRAME_COUNT Number of frames in the video file.
        // std::cout<<"id: "<<cap.get(cv::CAP_PROP_FRAME_COUNT)<<std::endl;
        
        if (!frame_loc.data)
        {
            usleep(1000000);
            cap.open(info_f->input);
            printf("cap reinitialize\n");
            continue;
        }

        end_t = std::chrono::steady_clock::now();
        std::cout << " VC-TIME 1 : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << " ms"<<std::endl;
        start_t = end_t;

        sem_vc.lock();
        info_f->frame = frame_loc.clone();
        info_f->frame_nbr = frame_nbr_loc;
        sem_vc.unlock();
        end_t = std::chrono::steady_clock::now();
        std::cout << " VC-TIME 2 : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << " ms"<<std::endl;
        start_t = end_t;
        frame_nbr_loc ++;
    }
}

/* Thread function to show the updated images 
**/
void *show_updates(void *x_void_ptr)
{
    cv::namedWindow("original", cv::WINDOW_NORMAL);
    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    cv::namedWindow("topview", cv::WINDOW_NORMAL);
    cv::namedWindow("disparity", cv::WINDOW_NORMAL);
    cv::Mat original_loc, detection_loc, topview_loc, disparity_loc;
    bool update_o_loc, update_de_loc, update_t_loc, update_di_loc;
    
    while (gRun)
    {
        TIMER_START
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        if(mutex_o.try_lock())
            {
            update_o_loc = updates.update_o;
            updates.update_o = false;
            if(update_o_loc)
                original_loc = updates.original.clone();
            mutex_o.unlock();
            }

        if(mutex_de.try_lock())
            {    
            update_de_loc = updates.update_de;
            updates.update_de = false;
            if(update_de_loc)
                detection_loc = updates.detection.clone();
            mutex_de.unlock();
            }

        if(mutex_t.try_lock())
            {  
            update_t_loc = updates.update_t;
            updates.update_t = false;
            if(update_t_loc)
                topview_loc = updates.topview.clone();
            mutex_t.unlock();
            }

        if(mutex_di.try_lock())
            {  
            update_di_loc = updates.update_di;
            updates.update_di = false;
            if(update_di_loc)
                disparity_loc = updates.disparity.clone();
            mutex_di.unlock();
            }

        if(update_o_loc)
            cv::imshow("original", original_loc);
        if(update_de_loc)
            cv::imshow("detection", detection_loc);
        if(update_t_loc)
            cv::imshow("topview", topview_loc);
        if(update_di_loc)
            cv::imshow("disparity", disparity_loc);
        cv::waitKey(1);
        // usleep(20000);  //sleep 20 msec
        std::cout<<"show_updates: ";
        TIMER_STOP
    }
}

void *originalFrame(void *x_void_ptr)
{
    Frame_t *info_show_orig = (Frame_t *) x_void_ptr;
    cv::Mat frame_loc;
    int frame_nbr_loc = 0;
    while (gRun)
    {
        TIMER_START
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        sem_vc.lock();
        frame_loc = info_show_orig->frame.clone();
        frame_nbr_loc = info_show_orig->frame_nbr;
        sem_vc.unlock();
        if (frame_nbr_loc == 0)
        {
            usleep(1000000);
            printf("no frame received\n");
            continue;
        }
        mutex_o.lock();
        updates.original = frame_loc.clone();
        updates.update_o = true;
        mutex_o.unlock();
        usleep(10000);  //sleep 10 msec
        std::cout<<"originalFrame: ";
        TIMER_STOP
    }
}

void *detectionFrame(void *x_void_ptr)
{
    ModFrame_t *info_show = (ModFrame_t *) x_void_ptr;
    double lat, lon, alt;
    int pix_x, pix_y;
    cv::Mat original_frame_loc;
    std::vector<Tracker> trackers;
    geodetic_converter::GeodeticConverter gc;
    double adfGeoTransform[6];
    cv::Mat H;
    tk::dnn::Yolo3Detection yolo;
    int num_detected;
    cv::Mat mask;

    // box variable
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;;
    float prob; 
    cv::Scalar intensity;

    while (gRun)
    {
        TIMER_START
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        sem.lock();
        original_frame_loc = info_show->original_frame.clone();
        // std::vector<Tracker> trackers;
        trackers = info_show->trackers;
        // geodetic_converter::GeodeticConverter gc;
        gc = info_show->gc;
        for(int i = 0; i < 6; i++ )
            adfGeoTransform[i] = info_show->adfGeoTransform[i];
        // cv::Mat H;
        H = info_show->H.clone();
        yolo = info_show->yolo;
        mask = info_show->mask.clone();
        sem.unlock();
        
        if (trackers.empty())
        {
            usleep(1000000);
            printf("no data available\n");
            continue;
        }

        num_detected = yolo.detected.size();
        for (int i = 0; i < num_detected; i++)
        {
            b = yolo.detected[i];
            x0 = b.x;
            w = b.w;
            x1 = b.x + w;
            y0 = b.y;
            h = b.h;
            y1 = b.y + h;
            objClass = b.cl;
            det_class = obj_class[b.cl];
            prob = b.prob;

            intensity = mask.at<uchar>(cv::Point(int(x0 + b.w / 2), y1));
            
            if (intensity[0] && objClass < 6)
            {
                //std::cout<<objClass<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
                cv::rectangle(original_frame_loc, cv::Point(x0, y0), cv::Point(x1, y1), yolo.colors[objClass], 2);
                // draw label
                int baseline = 0;
                float fontScale = 0.5;
                int thickness = 2;
                cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
                cv::rectangle(original_frame_loc, cv::Point(x0, y0), cv::Point((x0 + textSize.width - 2), (y0 - textSize.height - 2)), yolo.colors[b.cl], -1);
                cv::putText(original_frame_loc, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness); 
            }
        }

        for (auto t : trackers)
        {
            for (size_t p = 1; p < t.pred_list_.size(); p++)
            {
                gc.enu2Geodetic(t.pred_list_[p].x_, t.pred_list_[p].y_, 0, &lat, &lon, &alt);
                
                coord2pixel(lat, lon, pix_x, pix_y, adfGeoTransform);

                std::vector<cv::Point2f> map_p, camera_p;
                map_p.push_back(cv::Point2f(pix_x, pix_y));

                //transform camera pixel to map pixel
                cv::perspectiveTransform(map_p, camera_p, H.inv());
                cv::circle(original_frame_loc, cv::Point(camera_p[0].x, camera_p[0].y), 3.0, cv::Scalar(t.r_, t.g_, t.b_), CV_FILLED, 8, 0);
            }
        }

        mutex_de.lock();
        updates.detection = original_frame_loc.clone();
        updates.update_de = true;
        mutex_de.unlock();

        std::cout<<"detectionFrame: ";
        TIMER_STOP

    }
}

void *topviewFrame(void *x_void_ptr)
{
    ModFrame_t *info_show = (ModFrame_t *) x_void_ptr;
    double lat, lon, alt;
    int pix_x, pix_y;
    cv::Mat frame_top;
    cv::Mat original_frame_top;
    // original_frame_top = cv::imread("../demo/demo/data/map/map_geo.jpg");
    // original_frame_top = cv::imread("../demo/demo/data/map/MASA_4670.png");
    original_frame_top = cv::imread("../demo/demo/data/map/MASA_4670_V.png");
    
    std::vector<Tracker> trackers;
    geodetic_converter::GeodeticConverter gc;
    double adfGeoTransform[6];
    cv::Mat H;
    while (gRun)
    {
        TIMER_START
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        sem.lock();
        // std::vector<Tracker> trackers;
        trackers = info_show->trackers;
        // geodetic_converter::GeodeticConverter gc;
        gc = info_show->gc;
        for(int i = 0; i < 6; i++ )
            adfGeoTransform[i] = info_show->adfGeoTransform[i];
        // cv::Mat H;
        H = info_show->H.clone();
        sem.unlock();
        if (trackers.empty())
        {
            usleep(1000000);
            printf("no data available\n");
            continue;
        }
        frame_top = original_frame_top.clone();
        for (auto t : trackers)
        {
            for (size_t p = 1; p < t.pred_list_.size(); p++)
            {

                gc.enu2Geodetic(t.pred_list_[p].x_, t.pred_list_[p].y_, 0, &lat, &lon, &alt);
                //std::cout << "lat: " << lat << " lon: " << lon << std::endl;
                
                coord2pixel(lat, lon, pix_x, pix_y, adfGeoTransform);

                //std::cout << "pix_x: " << pix_x << " pix_y: " << pix_y << std::endl;
                if (pix_x < frame_top.cols && pix_y < frame_top.rows && pix_x >= 0 && pix_y >= 0)
                    cv::circle(frame_top, cv::Point(pix_x, pix_y), 7.0, cv::Scalar(t.r_, t.g_, t.b_), CV_FILLED, 8, 0);

            }
        }
        //outputVideo<< frame_top;
        // ------------------------------------------------
        mutex_t.lock();
        updates.topview = frame_top.clone();
        updates.update_t = true;
        mutex_t.unlock();

        std::cout<<"topviewFrame: ";
        TIMER_STOP

    }    
}

void *disparityFrame(void *x_void_ptr)
{
    Frame_t *info_show_disparity = (Frame_t *) x_void_ptr;
    bool first_iteration = true;
    cv::Mat frame_loc;
    int frame_nbr_loc = 0, pre_frame_nbr_loc = 0;
    auto start_t = std::chrono::steady_clock::now();
    auto step_t = std::chrono::steady_clock::now();
    auto end_t = std::chrono::steady_clock::now();

    // information for the disparity map
    cv::Mat canny, pre_canny, canny_RGB, pre_canny_RGB;
    cv::Mat canny_img;
    cv::Mat disparity_frame;
    while (gRun)
    {
        start_t = std::chrono::steady_clock::now();
        step_t = start_t;
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        sem_vc.lock();
        frame_loc = info_show_disparity->frame.clone();
        frame_nbr_loc = info_show_disparity->frame_nbr;
        sem_vc.unlock();
        if (frame_nbr_loc == 0)
        {
            usleep(1000000);
            printf("no frame received\n");
            continue;
        }
        // compute frame disparity only in there is a new frame
        if(frame_nbr_loc - pre_frame_nbr_loc > 0)
        {
            pre_frame_nbr_loc = frame_nbr_loc;
            //preprocessing frame
            step_t = std::chrono::steady_clock::now();
            // src_gray
            canny_img = img_laplacian(frame_loc,0);
            cv::Canny(canny_img, canny, 100, 100*2 );
            // sprintf(buf_frame_crop_name,"../demo/demo/data/img_disparity/%d_%d_canny.jpg",frame_nbr_loc, 999);
            // cv::imwrite(buf_frame_crop_name, canny);
            end_t = std::chrono::steady_clock::now();
            std::cout << " TIME END pre canny : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms"<<std::endl;
            step_t = end_t;
            // std::cout<<"o: "<<frame_loc.cols<<" - "<<frame_loc.rows<<std::endl;
            // std::cout<<"canny: "<<canny.cols<<" - "<<canny.rows<<std::endl;
            // std::cout<<"pre: "<<pre_canny.cols<<" - "<<pre_canny.rows<<std::endl;
            if(!first_iteration)
            {
                // backtorgb = cv::cvtColor(pre_canny,cv::COLOR_GRAY2RGB) 
                cv::cvtColor(pre_canny, pre_canny_RGB, CV_GRAY2RGB);
                cv::cvtColor(canny, canny_RGB, CV_GRAY2RGB);
                disparity_frame = frame_disparity(pre_canny_RGB, canny_RGB, frame_nbr_loc, 999, 0);
                // std::cout<<"size: "<<disparity_frame.rows<<" - "<<disparity_frame.cols<<std::endl;
                // if (disparity_frame.rows == 0 || disparity_frame.cols == 0)
                //     return -1;
                // if (disparity_frame.empty()) 
                // { // only fools don't check...
                //     std::cout << "image not loaded !" << std::endl;
                //     return -1;
                // }
                end_t = std::chrono::steady_clock::now();
                std::cout << " TIME canny : frame_disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms"<<std::endl;
                step_t = end_t;

                // //--------------------------------
                // //frame box disparity on the original image
                // step_t_segmentation = std::chrono::steady_clock::now();
                // frame_box_disparity(pre_frame, frame, pre_rois, frame_nbr_loc);
                // // reset pre_rois for the new roi of the current frame
                // // pre_rois.erase(pre_rois.begin(), pre_rois.end());
                // end_t_segmentation = std::chrono::steady_clock::now();
                // std::cout << " TIME Frame disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
                // step_t_segmentation = end_t_segmentation;

                // //frame box disparity on the preprocessed image
                // cv::cvtColor(pre_canny, pre_canny_RGB, CV_GRAY2RGB);
                // cv::cvtColor(canny, canny_RGB, CV_GRAY2RGB);
                // frame_box_disparity(pre_canny_RGB, canny_RGB, pre_rois, frame_nbr_loc);
                // // reset pre_rois for the new roi of the current frame
                // pre_rois.erase(pre_rois.begin(), pre_rois.end());
                // end_t_segmentation = std::chrono::steady_clock::now();
                // std::cout << " TIME Canny Frame disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
                // step_t_segmentation = end_t_segmentation;
                // //---------------------------------

                mutex_di.lock();
                updates.disparity = disparity_frame.clone();
                updates.update_di = true;
                mutex_di.unlock();
            }
            pre_canny = canny.clone();
            if(first_iteration)
                first_iteration = false;
            end_t = std::chrono::steady_clock::now();
            std::cout<<"disparityFrame : TIME END pre canny : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << " ms"<<std::endl;
        }
    }
}

int main(int argc, char *argv[])
{

    std::cout << "detection\n";
    signal(SIGINT, sig_handler);

    char *net = "yolo3_coco4.rt";
    if (argc > 1)
        net = argv[1];
    char *input = "../demo/demo/data/single_ped_2.mp4";
    if (argc > 2)
        input = argv[2];
    char *pmatrix = "../demo/demo/data/pmundist.txt";
    if (argc > 3)
        pmatrix = argv[3];
    char *tiffile = "../demo/demo/data/map_b.tif";
    if (argc > 4)
        tiffile = argv[4];
    int CAM_IDX = 20936;
    if (argc > 5)
        CAM_IDX = atoi(argv[5]);
    bool to_show = true;
    if (argc > 6)
        to_show = atoi(argv[6]);
    char *maskfile = "../demo/demo/data/mask36.jpg";
    if (argc > 7)
        maskfile = argv[7];
    char *cameraCalib = "../demo/demo/data/calib36.params";
    if (argc > 8)
        cameraCalib = argv[8];
    char *maskFileOrient = "../demo/demo/data/mask_orient/6315_mask_orient.jpg";
    if (argc > 9)
        maskFileOrient = argv[9];

    tk::dnn::Yolo3Detection yolo;
    yolo.init(net);
    yolo.thresh = 0.25;
    gRun = true;

    pthread_t visual, videocap, originalshow, detectionshow, topviewshow, disparityshow;
    //create video capture thread
    Frame_t info_f;
    info_f.input = input;
    if (pthread_create(&videocap, NULL, readVideoCapture, (void*)&info_f))
    {
        fprintf(stderr, "Error creating thread\n");
        return 1;
    };
      
     

    cv::Mat frame;
    cv::Mat frame_crop;
    char buf_frame_crop_name [200];
    cv::Mat dnn_input;

    /*projection matrix from camera to map*/
    cv::Mat H(cv::Size(3, 3), CV_64FC1);
    read_projection_matrix(H, pmatrix);

    /*Camera calibration*/
    cv::Mat cameraMat, distCoeff;
    readCameraCalibrationYaml(cameraCalib, cameraMat, distCoeff);
    std::cout << cameraMat << std::endl;
    std::cout << distCoeff << std::endl;

    /*GPS information*/
    double *adfGeoTransform = (double *)malloc(6 * sizeof(double));
    readTiff(tiffile, adfGeoTransform);
    std::vector<ObjCoords> coords;

    /*socket*/
    Communicator Comm(SOCK_DGRAM);
    Comm.open_client_socket("127.0.0.1", 8888);
    Message *m = new Message;
    m->cam_idx = CAM_IDX;
    m->lights.clear();

    /*Conversion for tracker, from gps to meters and viceversa*/
    geodetic_converter::GeodeticConverter gc;
    gc.initialiseReference(44.655540, 10.934315, 0);
    double east, north, up;
    // double lat, lon, alt;

    /*Mask info*/
    cv::Mat mask = cv::imread(maskfile, cv::IMREAD_GRAYSCALE);
    cv::Mat maskOrient = cv::imread(maskFileOrient);

    /*for(int i=0; i< mask.cols; i++)
    {
        for(int j=0; j< mask.rows; j++)
        {
            std::cout<<maskOrient.at<cv::Vec3b>(i,j)    <<std::endl;
        }
    }
    

    return 0;*/

    /*tracker infos*/
    srand(time(NULL));
    std::vector<Tracker> trackers;
    std::vector<Data> cur_frame;
    int initial_age = -5;
    int age_threshold = -8;
    int n_states = 5;
    float dt = 0.03;

    int frame_nbr = 0;

    //save video
    /*cv::VideoWriter outputVideo;
    cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), //Acquire input size
                          (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    outputVideo.open("test.avi", static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)), cap.get(cv::CAP_PROP_FPS), S, true);*/

    cv::Mat map1, map2;
    auto start_t = std::chrono::steady_clock::now();
    auto step_t = std::chrono::steady_clock::now();
    auto end_t = std::chrono::steady_clock::now();
    auto step_t_segmentation = std::chrono::steady_clock::now();
    auto end_t_segmentation = std::chrono::steady_clock::now();

    //TODO: move in a thread
    // // information for the disparity map
    std::vector <cv::Rect> pre_rois;
    // cv::Mat pre_frame;
    cv::Mat orig_frame;
    // cv::Mat canny, pre_canny, canny_RGB, pre_canny_RGB;
    // cv::Mat canny_img;

    // box variable
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;;
    float prob; 
    cv::Scalar intensity;
    cv::Rect roi;

    ModFrame_t info_show;

    if (to_show)
    {
        // initialize updates struct
        updates.update_o = false;
        updates.update_de = false;
        updates.update_t = false;
        updates.update_di = false;
        if (pthread_create(&visual, NULL, show_updates, (void*)NULL))
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        };
        if (pthread_create(&originalshow, NULL, originalFrame, (void*)&info_f))
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        };
        if (pthread_create(&disparityshow, NULL, disparityFrame, (void*)&info_f))
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        };
        info_show.H = cv::Mat(cv::Size(3, 3), CV_64FC1);
        if (pthread_create(&detectionshow, NULL, detectionFrame, (void*)&info_show))
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        };
        if (pthread_create(&topviewshow, NULL, topviewFrame, (void*)&info_show))
        {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        };
        // if (pthread_create(&detectionshow, NULL, showImages, (void *)&info_show))
        // {
        //     fprintf(stderr, "Error creating thread\n");
        //     return 1;
        // }
    } 
 

    bool first_iteration = true;
    while (gRun)
    {       
        TIMER_START
        start_t = std::chrono::steady_clock::now();
        step_t = start_t;
        
        sem_vc.lock();
        frame = info_f.frame.clone();
        if(info_f.frame_nbr - frame_nbr > 1)
            std::cout<<"more than one - f_n\n";
        frame_nbr = info_f.frame_nbr;
        sem_vc.unlock();
        std::cout<<"f_n: "<<frame_nbr<<std::endl;
        // if (!frame.data)
        if (frame_nbr == 0)
        {
            usleep(1000000);
            printf("no frame received\n");
            continue;
        }
        orig_frame = frame.clone();
        if (first_iteration)
            cv::initUndistortRectifyMap(cameraMat, distCoeff, cv::Mat(), cameraMat, frame.size(), CV_16SC2, map1, map2);
        cv::Mat temp = frame.clone();

        cv::remap(temp, frame, map1, map2, 1);
        
        //undistort(temp, frame, cameraMat, distCoeff);

        // this will be resized to the net format
        dnn_input = frame.clone();
        // TODO: async infer
        yolo.update(dnn_input);

        int num_detected = yolo.detected.size();
        if (num_detected > MAX_DETECT_SIZE)
            num_detected = MAX_DETECT_SIZE;

        coords.clear();

        end_t = std::chrono::steady_clock::now();
        std::cout << " TIME 1 : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms"<<std::endl;
        step_t = end_t;
        // draw dets
        std::cout<<"num detected: "<<num_detected<<std::endl;

        //TODO: move in a thread
        // //preprocessing frame
        // step_t_segmentation = std::chrono::steady_clock::now();
        // // src_gray
        // canny_img = img_laplacian(orig_frame,0);
        // cv::Canny(canny_img, canny, 100, 100*2 );
        // // sprintf(buf_frame_crop_name,"../demo/demo/data/img_disparity/%d_%d_canny.jpg",frame_nbr, 999);
        // // cv::imwrite(buf_frame_crop_name, canny);
        // end_t_segmentation = std::chrono::steady_clock::now();
        // std::cout << " - TIME END pre canny : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
        // step_t_segmentation = end_t_segmentation;
        // // std::cout<<"o: "<<orig_frame.cols<<" - "<<orig_frame.rows<<std::endl;
        // // std::cout<<"canny: "<<canny.cols<<" - "<<canny.rows<<std::endl;
        // // std::cout<<"pre: "<<pre_canny.cols<<" - "<<pre_canny.rows<<std::endl;
        // if(!first_iteration)
        // {
        //     // backtorgb = cv::cvtColor(pre_canny,cv::COLOR_GRAY2RGB) 
        //     cv::cvtColor(pre_canny, pre_canny_RGB, CV_GRAY2RGB);
        //     cv::cvtColor(canny, canny_RGB, CV_GRAY2RGB);
        //     disparity_frame = frame_disparity(pre_canny_RGB, canny_RGB, frame_nbr, 999, 0);
        //     std::cout<<"size: "<<disparity_frame.rows<<" - "<<disparity_frame.cols<<std::endl;
        //     if (disparity_frame.rows == 0 || disparity_frame.cols == 0)
        //         return -1;
        //     if (disparity_frame.empty()) 
        //     { // only fools don't check...
        //         std::cout << "image not loaded !" << std::endl;
        //         return -1;
        //     }
        //     end_t_segmentation = std::chrono::steady_clock::now();
        //     std::cout << " TIME canny : frame_disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
        //     step_t_segmentation = end_t_segmentation;

        //     // //--------------------------------
        //     // //frame box disparity on the original image
        //     // step_t_segmentation = std::chrono::steady_clock::now();
        //     // frame_box_disparity(pre_frame, frame, pre_rois, frame_nbr);
        //     // // reset pre_rois for the new roi of the current frame
        //     // // pre_rois.erase(pre_rois.begin(), pre_rois.end());
        //     // end_t_segmentation = std::chrono::steady_clock::now();
        //     // std::cout << " TIME Frame disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
        //     // step_t_segmentation = end_t_segmentation;

        //     // //frame box disparity on the preprocessed image
        //     // cv::cvtColor(pre_canny, pre_canny_RGB, CV_GRAY2RGB);
        //     // cv::cvtColor(canny, canny_RGB, CV_GRAY2RGB);
        //     // frame_box_disparity(pre_canny_RGB, canny_RGB, pre_rois, frame_nbr);
        //     // // reset pre_rois for the new roi of the current frame
        //     // pre_rois.erase(pre_rois.begin(), pre_rois.end());
        //     // end_t_segmentation = std::chrono::steady_clock::now();
        //     // std::cout << " TIME Canny Frame disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
        //     // step_t_segmentation = end_t_segmentation;
        //     // //---------------------------------
        // }

        // compute some metrics on the whole frame
        // segmentation(pre_frame, frame, frame_nbr, 0, 0);

        for (int i = 0; i < num_detected; i++)
        {
            b = yolo.detected[i];
            x0 = b.x;
            w = b.w;
            x1 = b.x + w;
            y0 = b.y;
            h = b.h;
            y1 = b.y + h;
            objClass = b.cl;
            det_class = obj_class[b.cl];
            prob = b.prob;

            intensity = mask.at<uchar>(cv::Point(int(x0 + b.w / 2), y1));
            
            if (intensity[0])
            {

                if (objClass < 6)
                {

                    // find the rectangular on the frame (sub-figure)                        
                    roi.x = (x0 > 0)? x0 : 0;
                    roi.y = (y0 > 0)? y0 : 0;
                    // std::cout<<"x "<<roi.x<<" - y "<<roi.y<<std::endl;
                    roi.width = (roi.x+w >= frame.cols)? frame.cols-1-roi.x : w;
                    roi.height = (roi.y+h >= frame.rows)? frame.rows-1-roi.y : h;
                    // std::cout<<"w "<<roi.width<<" - h "<<roi.height<<std::endl;
                    // std::cout<<"wf "<<frame.cols<<" - hf "<<frame.rows<<std::endl;
                    // std::cout<<"---"<<std::endl;
                    // std::cout<<"x "<<roi.x<<" to "<<roi.width+roi.x<<" wf "<<frame.cols<<std::endl;
                    // std::cout<<"y "<<roi.y<<" to "<<roi.height+roi.y<<" hf "<<frame.rows<<std::endl;
                    //update pre_roi for the next frame
                    pre_rois.push_back(roi);

                    // segmentation(frame(roi), frame(roi), frame_nbr, i, 1);

                    /////
                    convert_coords(coords, x0 + b.w / 2, y1, objClass, H, adfGeoTransform, frame_nbr);

                    // //std::cout<<objClass<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
                    // cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), yolo.colors[objClass], 2);
                    // // draw label
                    // int baseline = 0;
                    // float fontScale = 0.5;
                    // int thickness = 2;
                    // cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
                    // cv::rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + textSize.width - 2), (y0 - textSize.height - 2)), yolo.colors[b.cl], -1);
                    // cv::putText(frame, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness); 
                }
            }
        }

        // end_t = std::chrono::steady_clock::now();
        // std::cout << " TIME 2 : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms"<<std::endl;
        // step_t = end_t;
        //convert from latitude and longitude to meters for ekf
        cur_frame.clear();
        for (size_t i = 0; i < coords.size(); i++)
        {
            gc.geodetic2Enu(coords[i].lat_, coords[i].long_, 0, &east, &north, &up);
            cur_frame.push_back(Data(east, north, frame_nbr, coords[i].class_));
        }
        if (first_iteration)
        {
            for (auto f : cur_frame)
                trackers.push_back(Tracker(f, initial_age, dt, n_states));
        }
        else
        {
            Track(cur_frame, dt, n_states, initial_age, age_threshold, trackers);
        }

        std::cout << "There are " << trackers.size() << " trackers" << std::endl;

        //prepare message with tracker info
        addRoadUserfromTracker(trackers, m, gc, maskOrient, adfGeoTransform);
        //prepare the message with detection info
        //prepare_message(m, coords, CAM_IDX);
        //send message
        Comm.send_message(m);

        if (to_show)
        {
            //populate the ModFrame_t 
            sem.lock();
            info_show.original_frame = frame.clone();
            // std::vector<Tracker> trackers;
            info_show.trackers = trackers;
            // geodetic_converter::GeodeticConverter gc;
            info_show.gc = gc;
            for(int i = 0; i < 6; i++ )
                info_show.adfGeoTransform[i] = adfGeoTransform[i];
            // cv::Mat H;
            info_show.H = H.clone();
            info_show.yolo = yolo;
            info_show.mask = mask.clone();
            sem.unlock();
        }

        // update pre_frame for the disparity map
        // pre_frame = orig_frame.clone();
        // pre_canny = canny.clone();
        if(first_iteration)
            first_iteration = false;
    
        frame_nbr++;
        std::cout<<"MAIN thread: ";
        TIMER_STOP
    }

    free(adfGeoTransform);

    std::cout << "detection end\n";
    return 0;
}
