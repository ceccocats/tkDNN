#include <iostream>
#include <signal.h>
#include <stdlib.h> /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <ctime>
#include <pthread.h>

#include "utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

cv::Mat frame_v;
cv::Mat frame_top_v;
std::mutex sem;

void sig_handler(int signo)
{
    std::cout << "request gateway stop\n";
    gRun = false;
}

void *showImages(void *x_void_ptr)
{
    while (gRun)
    {
        sem.lock();
        cv::imshow("detection", frame_v);
        cv::imshow("topview", frame_top_v);
        sem.unlock();
        cv::waitKey(30);
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

    tk::dnn::Yolo3Detection yolo;
    yolo.init(net);
    yolo.thresh = 0.25;

    gRun = true;

    cv::VideoCapture cap(input);
    if (!cap.isOpened())
        gRun = false;
    else
        std::cout << "camera started\n";

    cv::Mat frame;
    cv::Mat frame_top;
    cv::Mat dnn_input;
    cv::Mat original_frame_top;

    pthread_t visual;

    if (to_show)
    {
        cv::namedWindow("detection", cv::WINDOW_NORMAL);
        cv::namedWindow("topview", cv::WINDOW_NORMAL);
        frame_top = cv::imread("../demo/demo/data/map/map_geo.jpg");
        original_frame_top = frame_top.clone();
    }

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
    double lat, lon, alt;

    /*Mask info*/
    cv::Mat mask = cv::imread(maskfile, cv::IMREAD_GRAYSCALE);

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

    while (gRun)
    {

        cap >> frame;

        cv::Mat temp = frame.clone();
        undistort(temp, frame, cameraMat, distCoeff);
        cv::imwrite(std::to_string(CAM_IDX) + ".jpg", frame);

        if (!frame.data)
        {
            usleep(1000000);
            cap.open(input);
            printf("cap reinitialize\n");
            continue;
        }

        // this will be resized to the net format
        dnn_input = frame.clone();
        // TODO: async infer
        yolo.update(dnn_input);

        int num_detected = yolo.detected.size();
        if (num_detected > MAX_DETECT_SIZE)
            num_detected = MAX_DETECT_SIZE;

        coords.clear();

        // draw dets
        for (int i = 0; i < num_detected; i++)
        {

            tk::dnn::box b = yolo.detected[i];
            int x0 = b.x;
            int x1 = b.x + b.w;
            int y0 = b.y;
            int y1 = b.y + b.h;
            int objClass = b.cl;
            std::string det_class = obj_class[b.cl];
            float prob = b.prob;

            cv::Scalar intensity = mask.at<uchar>(cv::Point(int(x0 + b.w / 2), y1));

            if (intensity[0])
            {

                if (objClass < 6)
                {
                    convert_coords(coords, x0 + b.w / 2, y1, objClass, H, adfGeoTransform, frame_nbr);

                    //std::cout<<objClass<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
                    cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), yolo.colors[objClass], 2);
                    // draw label
                    int baseline = 0;
                    float fontScale = 0.5;
                    int thickness = 2;
                    cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
                    cv::rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + textSize.width - 2), (y0 - textSize.height - 2)), yolo.colors[b.cl], -1);
                    cv::putText(frame, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
                }
            }
        }

        TIMER_START

        //convert from latitude and longitude to meters for ekf
        cur_frame.clear();
        for (size_t i = 0; i < coords.size(); i++)
        {
            gc.geodetic2Enu(coords[i].lat_, coords[i].long_, 0, &east, &north, &up);
            cur_frame.push_back(Data(east, north, frame_nbr, coords[i].class_));
        }
        if (frame_nbr == 0)
        {
            for (auto f : cur_frame)
                trackers.push_back(Tracker(f, initial_age, dt, n_states));
        }
        else
        {
            Track(cur_frame, dt, n_states, initial_age, age_threshold, trackers);
        }

        TIMER_STOP
        std::cout << "There are " << trackers.size() << " trackers" << std::endl;


        //prepare message with tracker info
        addRoadUserfromTracker(trackers, m, gc);
        //prepare the message with detection info
        //prepare_message(m, coords, CAM_IDX);
        //send message
        Comm.send_message(m);


        if (to_show)
        {
            frame_top = original_frame_top.clone();
            for (auto t : trackers)
            {
                for (size_t p = 1; p < t.pred_list_.size(); p++)
                {

                    gc.enu2Geodetic(t.pred_list_[p].x_, t.pred_list_[p].y_, 0, &lat, &lon, &alt);
                    //std::cout << "lat: " << lat << " lon: " << lon << std::endl;
                    int pix_x, pix_y;
                    coord2pixel(lat, lon, pix_x, pix_y, adfGeoTransform);

                    //std::cout << "pix_x: " << pix_x << " pix_y: " << pix_y << std::endl;
                    if (pix_x < frame_top.cols && pix_y < frame_top.rows && pix_x >= 0 && pix_y >= 0)
                        cv::circle(frame_top, cv::Point(pix_x, pix_y), 7.0, cv::Scalar(t.r_, t.g_, t.b_), CV_FILLED, 8, 0);

                    std::vector<cv::Point2f> map_p, camera_p;
                    map_p.push_back(cv::Point2f(pix_x, pix_y));

                    //transform camera pixel to map pixel
                    cv::perspectiveTransform(map_p, camera_p, H.inv());
                    //std::cout << "pix_x: " << camera_p[0].x << " pix_y: " << camera_p[0].y << std::endl;

                    cv::circle(frame, cv::Point(camera_p[0].x, camera_p[0].y), 3.0, cv::Scalar(t.r_, t.g_, t.b_), CV_FILLED, 8, 0);

                    /*if(p == t.pred_list_.size()-1)
                    {
                        auto center = cv::Point(camera_p[0].x, camera_p[0].y);
                        auto color = cv::Scalar(t.r_, t.g_, t.b_);
                        draw_arrow(t.pred_list_[p].yaw_, t.pred_list_[p].vel_,color, center, frame);

                        center = cv::Point(pix_x,pix_y);
                        draw_arrow(t.pred_list_[p].yaw_, t.pred_list_[p].vel_,color, center, frame_top);
                    }*/
                }
            }
        }

        if (to_show)
        {
            sem.lock();
            frame_v = frame.clone();
            frame_top_v = frame_top.clone();
            sem.unlock();

            //outputVideo<< frame_top;
        }

        if (frame_nbr == 0 && to_show)
        {
            if (pthread_create(&visual, NULL, showImages, NULL))
            {
                fprintf(stderr, "Error creating thread\n");
                return 1;
            }
        }

        frame_nbr++;
        
    }

    free(adfGeoTransform);

    std::cout << "detection end\n";
    return 0;
}
