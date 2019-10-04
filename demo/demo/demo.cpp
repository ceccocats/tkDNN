#include <time.h>

#include "utils.h"
#include "Yolo3Detection.h"
#include "message.h"
#include "visualization.h"

#include "tracker.h"
#include "../masa_protocol/include/send.hpp"
#include "../masa_protocol/include/serialize.hpp"

// #include <assert.h>
// #include <unistd.h>
// #include <mutex>
// #include <ctime>
// #include <pthread.h>
// #include <signal.h>
// #include <chrono>
// #include <math.h>
// #include <typeinfo>
// #include <iostream>

#define MAX_DETECT_SIZE 100

bool gRun;
std::chrono::steady_clock::time_point local_clock_start;
std::mutex mutexgRun;
std::string obj_class[10]{"person", "car", "truck", "bus", "motor", "bike", "rider", "traffic light", "traffic sign", "train"};
//mutex for some opencv operations
std::mutex mutex_cv;
Show_t updates;

void sig_handler(int signo)
{
    std::cout << "request gateway stop\n";
    mutexgRun.lock();
    gRun = false;
    mutexgRun.unlock();
}

void *readVideoCapture(void *x_void_ptr)
{
    std::cout << "readVideoCapture start...\n";

    Frame_t *info_f = (Frame_t *)x_void_ptr;
    mutex_cv.lock();
    cv::VideoCapture cap(info_f->input, cv::CAP_FFMPEG);
    mutex_cv.unlock();
    cv::Mat frame_loc, frame0;
    int frame_nbr_loc = 0;
    // bool to_show = false;
    if (!cap.isOpened())
    {
        mutexgRun.lock();
        gRun = false;
        mutexgRun.unlock();
    }
    else
        std::cout << "camera started\n";

    // cap.set(cv::CAP_PROP_BUFFERSIZE,3);
    // std::cout<<"buf size: "<<cap.get(CV_CAP_PROP_BUFFERSIZE)<<std::endl;
    auto start_t = std::chrono::steady_clock::now();
    auto step_t = std::chrono::steady_clock::now();
    auto end_t = std::chrono::steady_clock::now();
    auto current_timestamp = std::chrono::steady_clock::now();

    // compute fps and find camera's clock
    double shift, mean_time = 0;
    std::cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << cap.get(CV_CAP_PROP_FPS) << std::endl;
    std::cout << "readVideoCapture computes frame rate...\n";
    // //compute frame rate
    int i = 0;
    int num_f = 120;
    // the first 20 frames are null
    while (i < 21)
    {
        cap >> frame_loc;
        i++;
    }

    i = 0;
    start_t = std::chrono::steady_clock::now();
    while (i < num_f)
    {
        step_t = std::chrono::steady_clock::now();
        cap >> frame_loc;
        mean_time = mean_time + std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - step_t).count();
        std::cout << " step " << i << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - step_t).count() << " ms" << std::endl;
        i++;
    }
    end_t = std::chrono::steady_clock::now();

    std::cout << "Capturing " << num_f << " frames" << std::endl;
    std::cout << " Time taken : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << " ms" << std::endl;

    /*
    mean_time indicates the milliseconds from a frame and the next. (frame rate)
    local_clock_sync is the camera clock.
    shift is the difference from local camera clock and local process clock.
    a frame is allowed if its local timestamp minus its local clock is less then a tollerance, 
    otherwise it will be considered old.
    */
    auto local_clock_sync = std::chrono::steady_clock::now();
    mean_time = mean_time / num_f;
    shift = ((double)std::chrono::duration_cast<std::chrono::milliseconds>(local_clock_sync - local_clock_start).count()) / mean_time;
    shift = (shift - (int)shift) * mean_time;
    std::cout << ".-------------------------------\n";
    std::cout << " mean time: " << mean_time << std::endl;
    std::cout << " shift: " << shift << std::endl;
    std::cout << " TIMEDIFFERENCE:  " << std::chrono::duration_cast<std::chrono::milliseconds>(local_clock_sync - local_clock_start).count() << std::endl;
    std::cout << "\n\n\n\n";
    std::cout << "readVideoCapture start to capture...\n";
    while (gRun)
    {
        // mutex_cv.lock();
        cap >> frame_loc;
        // mutex_cv.unlock();
        current_timestamp = std::chrono::steady_clock::now();
        shift = std::chrono::duration_cast<std::chrono::milliseconds>(current_timestamp - local_clock_sync).count();
        std::cout << " RELATIVE TIMESTAMP FRAME : " << shift << " ms" << std::endl;
        shift = shift / mean_time;
        shift = (shift - (int)shift) * mean_time;
        shift = (shift - mean_time / 2 >= 0) ? -(mean_time - shift) : shift;
        std::cout << "DELAY frame_" << frame_nbr_loc << " : " << shift << " ms" << std::endl;
        // TODO: here introduce a tollerance to discard old frame

        // std::cout<< "CV_CAP_PROP_POS_MSEC:   "<< cap.get( cv::CAP_PROP_POS_MSEC) <<std::endl;
        // std::cout<< "CV_CAP_PROP_POS_FRAMES:  "<< cap.get( cv::CAP_PROP_POS_FRAMES) <<std::endl;  // <-- the v4l2 'sequence' field
        // std::cout<< "CV_CAP_PROP_FPS:  "<< cap.get( cv::CAP_PROP_FPS)<<std::endl;
        // std::cout << "Format: " << cap.get(CV_CAP_PROP_FORMAT) << "\n";
        // CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
        std::cout << "id: " << cap.get(cv::CAP_PROP_POS_MSEC) << std::endl;
        // CAP_PROP_FRAME_COUNT Number of frames in the video file.
        std::cout << "id: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;

        if (!frame_loc.data)
        {
            usleep(1000000);
            mutex_cv.lock();
            cap.open(info_f->input);
            printf("cap reinitialize\n");
            mutex_cv.unlock();
            continue;
        }

        end_t = std::chrono::steady_clock::now();
        std::cout << " VC-TIME 1 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << " ms" << std::endl;
        start_t = end_t;

        info_f->sem_vc.lock();
        info_f->frame = frame_loc.clone();
        info_f->frame_nbr = frame_nbr_loc;
        info_f->sem_vc.unlock();
        // usleep(50000);
        end_t = std::chrono::steady_clock::now();
        std::cout << " VC-TIME 2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << " ms" << std::endl;
        start_t = end_t;
        frame_nbr_loc++;
    }
    return (void *)0;
}

void *computationTask(void *x_void_ptr)
{
    Camera_t *camera = (Camera_t *)x_void_ptr;
    pthread_t visual, originalshow, detectionshow, topviewshow, disparityshow;
    pthread_t videocap;
    //create video capture thread
    Frame_t info_f;
    info_f.input = camera->input;
    if (pthread_create(&videocap, NULL, readVideoCapture, (void *)&info_f))
    {
        fprintf(stderr, "Error creating thread\n");
        return (void *)1;
    };

    bool to_show = camera->to_show;
    double adfGeoTransform[6];
    for (int i = 0; i < 6; i++)
        adfGeoTransform[i] = camera->adfGeoTransform[i];

    ModFrame_t info_show;
    if (to_show)
    {
        // initialize updates struct
        updates.update_o = false;
        updates.update_de = false;
        updates.update_t = false;
        updates.update_di = false;
        if (pthread_create(&visual, NULL, show_updates, (void *)NULL))
        {
            fprintf(stderr, "Error creating thread\n");
            return (void *)1;
        };
        if (pthread_create(&originalshow, NULL, originalFrame, (void *)&info_f))
        {
            fprintf(stderr, "Error creating thread\n");
            return (void *)1;
        };
        if (pthread_create(&disparityshow, NULL, disparityFrame, (void *)&info_f))
        {
            fprintf(stderr, "Error creating thread\n");
            return (void *)1;
        };
        info_show.H = cv::Mat(cv::Size(3, 3), CV_64FC1);
        if (pthread_create(&detectionshow, NULL, detectionFrame, (void *)&info_show))
        {
            fprintf(stderr, "Error creating thread\n");
            return (void *)1;
        };
        if (pthread_create(&topviewshow, NULL, topviewFrame, (void *)&info_show))
        {
            fprintf(stderr, "Error creating thread\n");
            return (void *)1;
        };
    }

    char *pmatrix = camera->pmatrix;
    /*projection matrix from camera to map*/
    cv::Mat H(cv::Size(3, 3), CV_64FC1);
    read_projection_matrix(H, pmatrix);
    assert(cv::countNonZero(H) > 0);
    // std::cout<<H<<std::endl;
    // return (void*)0;
    /*Camera calibration*/
    cv::Mat cameraMat, distCoeff;
    readCameraCalibrationYaml(camera->cameraCalib, cameraMat, distCoeff);
    std::cout << cameraMat << std::endl;
    std::cout << distCoeff << std::endl;

    /*GPS information*/
    std::vector<ObjCoords> coords;

    /*socket*/
    Communicator Comm(SOCK_DGRAM);
    Comm.open_client_socket((char *)"127.0.0.1", 8888);

    Message *m = new Message;
    m->cam_idx = camera->CAM_IDX;
    m->lights.clear();
    /*Conversion for tracker, from gps to meters and viceversa*/
    // mutex_cv.lock();
    geodetic_converter::GeodeticConverter gc;
    gc.initialiseReference(44.655540, 10.934315, 0);
    // mutex_cv.unlock();
    double east, north, up;
    // double lat, lon, alt;
    /*Mask info*/
    cv::Mat mask = cv::imread(camera->maskfile, cv::IMREAD_GRAYSCALE);
    cv::Mat maskOrient = cv::imread(camera->maskFileOrient);
    // cv::Mat maskOrient = cv::imread(camera->maskFileOrient, 0);

    /*for(int i=0; i< mask.cols; i++)
    {
        for(int j=0; j< mask.rows; j++)
        {
            std::cout<<maskOrient.at<cv::Vec3b>(i,j)    <<std::endl;
        }
    }
    
   
    return 0;*/
    /*tracker infos*/
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
    // auto step_t_segmentation = std::chrono::steady_clock::now();
    // auto end_t_segmentation = std::chrono::steady_clock::now();

    //TODO: move in a thread
    // // information for the disparity map
    // std::vector <cv::Rect> pre_rois;
    // cv::Mat pre_frame;
    cv::Mat orig_frame;
    // cv::Mat canny, pre_canny, canny_RGB, pre_canny_RGB;
    // cv::Mat canny_img;

    // box variable
    tk::dnn::box b;
    int x0, h, y1; //w, x1, y0;
    int objClass;
    std::string det_class;
    ;
    // float prob;
    cv::Scalar intensity;

    cv::Mat frame;
    cv::Mat frame_crop;
    cv::Mat dnn_input;
    bool first_iteration = true;
    while (gRun)
    {
        TIMER_START
        start_t = std::chrono::steady_clock::now();
        step_t = start_t;

        info_f.sem_vc.lock();
        frame = info_f.frame.clone();
        if (info_f.frame_nbr - frame_nbr > 1)
            std::cout << "more than one - f_n (diff " << info_f.frame_nbr - frame_nbr << ")\n";
        frame_nbr = info_f.frame_nbr;
        info_f.sem_vc.unlock();
        std::cout << "f_n: " << frame_nbr << std::endl;
        // if (!frame.data)
        if (frame_nbr == 0)
        {
            usleep(1000000);
            printf("no frame received\n");
            continue;
        }
        orig_frame = frame.clone();
        // mutex_cv.lock();
        if (first_iteration)
            cv::initUndistortRectifyMap(cameraMat, distCoeff, cv::Mat(), cameraMat, frame.size(), CV_16SC2, map1, map2);
        cv::Mat temp = frame.clone();
        cv::remap(temp, frame, map1, map2, 1);
        //undistort(temp, frame, cameraMat, distCoeff);
        // mutex_cv.unlock();

        // this will be resized to the net format
        dnn_input = frame.clone();
        // TODO: async infer
        camera->yolo.update(dnn_input);
        int num_detected = camera->yolo.detected.size();
        if (num_detected > MAX_DETECT_SIZE)
            num_detected = MAX_DETECT_SIZE;

        coords.clear();
        end_t = std::chrono::steady_clock::now();
        std::cout << " TIME 1 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
        step_t = end_t;
        // draw dets
        std::cout << "camera: " << camera->CAM_IDX << " - num detected: " << num_detected << std::endl;

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
            b = camera->yolo.detected[i];
            x0 = b.x;
            // w = b.w;
            // x1 = b.x + w;
            // y0 = b.y;
            h = b.h;
            y1 = b.y + h;
            objClass = b.cl;
            det_class = obj_class[b.cl];
            // prob = b.prob;

            intensity = mask.at<uchar>(cv::Point(int(x0 + b.w / 2), y1));

            if (intensity[0])
            {

                if (objClass < 6)
                {

                    // find the rectangular on the frame (sub-figure)
                    // roi.x = (x0 > 0)? x0 : 0;
                    // roi.y = (y0 > 0)? y0 : 0;
                    // // std::cout<<"x "<<roi.x<<" - y "<<roi.y<<std::endl;
                    // roi.width = (roi.x+w >= frame.cols)? frame.cols-1-roi.x : w;
                    // roi.height = (roi.y+h >= frame.rows)? frame.rows-1-roi.y : h;
                    // std::cout<<"w "<<roi.width<<" - h "<<roi.height<<std::endl;
                    // std::cout<<"wf "<<frame.cols<<" - hf "<<frame.rows<<std::endl;
                    // std::cout<<"---"<<std::endl;
                    // std::cout<<"x "<<roi.x<<" to "<<roi.width+roi.x<<" wf "<<frame.cols<<std::endl;
                    // std::cout<<"y "<<roi.y<<" to "<<roi.height+roi.y<<" hf "<<frame.rows<<std::endl;
                    //update pre_roi for the next frame
                    // pre_rois.push_back(roi);

                    // segmentation(frame(roi), frame(roi), frame_nbr, i, 1);

                    /////
                    convert_coords(coords, x0 + b.w / 2, y1, objClass, H, adfGeoTransform);

                    // //std::cout<<objClass<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
                    // cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), camera->yolo.colors[objClass], 2);
                    // // draw label
                    // int baseline = 0;
                    // float fontScale = 0.5;
                    // int thickness = 2;
                    // cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
                    // cv::rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + textSize.width - 2), (y0 - textSize.height - 2)), camera->yolo.colors[b.cl], -1);
                    // cv::putText(frame, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
                }
            }
        }

        end_t = std::chrono::steady_clock::now();
        std::cout << " TIME 2 : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
        step_t = end_t;
        //convert from latitude and longitude to meters for ekf
        cur_frame.clear();
        for (size_t i = 0; i < coords.size(); i++)
        {
            gc.geodetic2Enu(coords[i].lat_, coords[i].long_, 0, &east, &north, &up);
            cur_frame.push_back(Data(east, north, frame_nbr, coords[i].class_));
        }
        if (first_iteration)
        {
            // if there aren't detections and it is the first iteration, we can't initialize the tracker, so continue
            if (cur_frame.empty())
                continue;
            for (auto f : cur_frame)
                trackers.push_back(Tracker(f, initial_age, dt, n_states));
        }
        else
        {
            Track(cur_frame, dt, n_states, initial_age, age_threshold, trackers);
        }
        std::cout << "There are " << trackers.size() << " trackers" << std::endl;
        //prepare message with tracker info
        if (trackers.size() == 0)
        {
            // mutex_cv.lock();
            addRoadUserfromTracker(trackers, m, gc, maskOrient, adfGeoTransform, H);
            // mutex_cv.unlock();
            //prepare the message with detection info
            //prepare_message(m, coords, CAM_IDX);
            //send message
            if (!m->objects.empty())
                Comm.send_message(m);
        }

        if (to_show)
        {
            //populate the ModFrame_t
            info_show.sem.lock();
            info_show.original_frame = frame.clone();
            // std::vector<Tracker> trackers;
            info_show.trackers = trackers;
            // geodetic_converter::GeodeticConverter gc;
            info_show.gc = gc;
            for (int i = 0; i < 6; i++)
                info_show.adfGeoTransform[i] = adfGeoTransform[i];
            // cv::Mat H;
            info_show.H = H.clone();
            info_show.yolo = camera->yolo;
            // std::copy(camera->yolo.begin(), camera->yolo.end(), info_show.yolo.begin());
            info_show.mask = mask.clone();
            info_show.sem.unlock();
        }

        // update pre_frame for the disparity map
        // pre_frame = orig_frame.clone();
        // pre_canny = canny.clone();
        if (first_iteration)
            first_iteration = false;

        frame_nbr++;
        std::cout << camera->CAM_IDX << " camera thread: ";
        TIMER_STOP
    }
    return (void *)0;
}

int main(int argc, char *argv[])
{

    std::cout << "detection\n";
    signal(SIGINT, sig_handler);
    srand(time(NULL));
    bool no_params = false; //flag to indicate if there are camera parameters or we use mp4 test video

    char *net = (char *)"yolo3_coco4.rt";
    if (argc > 1)
        net = argv[1];
    char *tiffile = (char *)"../demo/demo/data/map_b.tif";
    if (argc > 2)
        tiffile = argv[2];
    char *n;
    if (argc > 3)
    {
        n = argv[3];
        if (strcmp(n, "-n"))
            return -1;
    };
    int n_cameras = 0;
    if (argc > 4)
    {
        n_cameras = atoi(argv[4]);
        if (argc < 5 + 7 * n_cameras)
        {
            std::cout << "too few parameters\n";
            return -1;
        }
        if (!n_cameras)
        {
            n_cameras = 1;
            no_params = true;
        }
    }
    Camera_t cameras[n_cameras];
    bool *to_show = (bool *)malloc(n_cameras * sizeof(bool));
    if (no_params)
    {
        cameras[0].CAM_IDX = 20936;
        cameras[0].input = (char *)"../demo/demo/data/single_ped_2.mp4";
        cameras[0].pmatrix = (char *)"../demo/demo/data/pmundist.txt";
        cameras[0].maskfile = (char *)"../demo/demo/data/mask36.jpg";
        cameras[0].cameraCalib = (char *)"../demo/demo/data/calib36.params";
        cameras[0].maskFileOrient = (char *)"../demo/demo/data/mask_orient/6315_mask_orient.jpg";
        cameras[0].to_show = true;
        to_show[0] = true;
    }
    else
    {
        for (int i = 0; i < n_cameras; i++)
            cameras[i].CAM_IDX = atoi(argv[5 + i]);
        for (int i = 0; i < n_cameras; i++)
            cameras[i].input = argv[5 + n_cameras + i];
        for (int i = 0; i < n_cameras; i++)
            cameras[i].pmatrix = argv[5 + 2 * n_cameras + i];
        for (int i = 0; i < n_cameras; i++)
            cameras[i].maskfile = argv[5 + 3 * n_cameras + i];
        for (int i = 0; i < n_cameras; i++)
            cameras[i].cameraCalib = argv[5 + 4 * n_cameras + i];
        for (int i = 0; i < n_cameras; i++)
            cameras[i].maskFileOrient = argv[5 + 5 * n_cameras + i];
        for (int i = 0; i < n_cameras; i++)
        {
            cameras[i].to_show = atoi(argv[5 + 6 * n_cameras + i]); //only one camera can be shown
            to_show[i] = atoi(argv[5 + 6 * n_cameras + i]);
        }
    }
    //TODO now only one camera can be visualized
    int check_visualization = 0;
    for (int i = 0; i < n_cameras; i++)
        check_visualization += to_show[i];
    if (check_visualization > 1)
        return -1;

    tk::dnn::Yolo3Detection yolo[n_cameras];
    for (int i = 0; i < n_cameras; i++)
    {
        yolo[i].init(net);
        yolo[i].thresh = 0.25;
    }
    // tk::dnn::Yolo3Detection yolo;
    // yolo.init(net);
    // yolo.thresh = 0.25;

    gRun = true;

    // start the local clock. It is used to check the incoming frames (by different cameras)
    local_clock_start = std::chrono::steady_clock::now();

    /*GPS information*/
    double *adfGeoTransform = (double *)malloc(6 * sizeof(double));
    readTiff(tiffile, adfGeoTransform);

    for (int i = 0; i < n_cameras; i++)
    {
        for (int j = 0; j < 6; j++)
            cameras[i].adfGeoTransform[j] = adfGeoTransform[j];
        cameras[i].yolo = yolo[i];
        // cameras[i].yolo = yolo;
    }
    std::cout << "INIZIA:\n";
    pthread_t camera_task[n_cameras];
    for (int i = 0; i < n_cameras; i++)
    {
        std::cout << "creating thread\n";
        if (pthread_create(&camera_task[i], NULL, computationTask, (void *)&cameras[i]))
        {
            fprintf(stderr, "error creating thread\n");
            return 1;
        }
        std::cout << "WHAT2\n";
    }
    for (int i = 0; i < n_cameras; i++)
    {
        pthread_join(camera_task[i], NULL);
    }
    std::cout << " free adfGeoT \n";
    free(adfGeoTransform);
    std::cout << "detection end\n";
    return 0;
}
