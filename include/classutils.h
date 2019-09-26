#ifndef CLASSUTILS_H
#define CLASSUTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/socket.h> //socket
#include <arpa/inet.h>  //inet_addr
#include <unistd.h>     //write

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include "gdal.h"
#include <gdal_priv.h>
#include <gdal/gdal.h>
#include "gdal/gdal_priv.h"
#include "gdal/cpl_conv.h"

#include "tracker.h"

#include <yaml-cpp/yaml.h>

#include "../masa_protocol/include/send.hpp"
#include "../masa_protocol/include/serialize.hpp"

struct ObjCoords
{
    double lat_;
    double long_;
    int class_;
};

void readTiff(char *filename, double *adfGeoTransform)
{
    GDALDataset *poDataset;
    GDALAllRegister();
    poDataset = (GDALDataset *)GDALOpen(filename, GA_ReadOnly);
    if (poDataset != NULL)
    {
        poDataset->GetGeoTransform(adfGeoTransform);
    }
}

void readCameraCalibrationYaml(const std::string &cameraCalib, cv::Mat &cameraMat, cv::Mat &distCoeff)
{
    YAML::Node config = YAML::LoadFile(cameraCalib);
    const YAML::Node &node_test1 = config["camera_matrix"];

    float data_cm[9];
    for (std::size_t i = 0; i < node_test1["data"].size(); i++)
        data_cm[i] = node_test1["data"][i].as<float>();
    cv::Mat cameraMat_ = cv::Mat(3, 3, CV_32F, data_cm);
    cameraMat = cameraMat_.clone();
    std::cout << cameraMat << std::endl;
    const YAML::Node &node_test2 = config["distortion_coefficients"];

    float data_dc[5];
    for (std::size_t i = 0; i < node_test2["data"].size(); i++)
        data_dc[i] = node_test2["data"][i].as<float>();
    cv::Mat distCoeff_ = cv::Mat(5, 1, CV_32F, data_dc);
    distCoeff = distCoeff_.clone();
    std::cout << distCoeff << std::endl;
}

void pixel2coord(int x, int y, double &lat, double &lon, double *adfGeoTransform)
{
    //Returns global coordinates from pixel x, y coordinates
    double xoff, a, b, yoff, d, e;
    xoff = adfGeoTransform[0];
    a = adfGeoTransform[1];
    b = adfGeoTransform[2];
    yoff = adfGeoTransform[3];
    d = adfGeoTransform[4];
    e = adfGeoTransform[5];

    //printf("%f %f %f %f %f %f\n",xoff, a, b, yoff, d, e );

    lon = a * x + b * y + xoff;
    lat = d * x + e * y + yoff;
}
void coord2pixel(double lat, double lon, int &x, int &y, double *adfGeoTransform)
{
    x = int(round((lon - adfGeoTransform[0]) / adfGeoTransform[1]));
    y = int(round((lat - adfGeoTransform[3]) / adfGeoTransform[5]));
}

void fillMatrix(cv::Mat &H, double *matrix, bool show = false)
{
    double *vals = (double *)H.data;
    for (int i = 0; i < 9; i++)
    {
        vals[i] = matrix[i];
    }
    if (show)
        std::cout << H << "\n";
}

//FILE *out_file = fopen("prova_pixel.txt", "w");

void convert_coords(std::vector<ObjCoords> &coords, int x, int y, int detected_class, cv::Mat H, double *adfGeoTransform, int frame_nbr)
{
    double latitude, longitude;
    std::vector<cv::Point2f> x_y, ll;
    x_y.push_back(cv::Point2f(x, y));
    //transform camera pixel to map pixel
    cv::perspectiveTransform(x_y, ll, H);
    //tranform to map pixel to map gps
    pixel2coord(ll[0].x, ll[0].y, latitude, longitude, adfGeoTransform);
    //printf("lat: %f, long:%f \n", latitude, longitude);

    ObjCoords coord;
    coord.lat_ = latitude;
    coord.long_ = longitude;
    coord.class_ = detected_class;
    coords.push_back(coord);

    /*if (detected_class == 0)
    {

        struct timeval tv;
        gettimeofday(&tv, NULL);
        unsigned long long t_stamp_ms = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;

        //printf(out_file, "%d %lld %d %d\n",frame_nbr, t_stamp_ms, int(ll[0].x), int(ll[0].y));
        fprintf(out_file, "%d %lld %f %f\n", frame_nbr, t_stamp_ms, coord.LAT, coord.LONG);
        //printf( "%d %lld %f %f\n", frame_nbr, t_stamp_ms, coord.LAT, coord.LONG);
    }*/
}

void read_projection_matrix(cv::Mat &H, char *path)
{
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    // float *proj_matrix = (float *)malloc(9 * sizeof(float));
    double proj_matrix[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int i = 0;
    fp = fopen(path, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
        
    while ((read = getline(&line, &len, fp)) != -1)
    {
        std::cout<<line<<std::endl;
        std::stringstream ss(line);
        while (ss >> proj_matrix[i])
            i++;
    }
    fclose(fp);
    fillMatrix(H, proj_matrix);
    free(line);
    // free(proj_matrix);
}

void draw_arrow(float angleRad, float vel, cv::Scalar color, cv::Point center, cv::Mat &frame)
{
    int angle = angleRad * 180.0 / CV_PI;
    auto length = 10 * vel;
    auto direction = cv::Point(length * cos(angleRad), length * sin(angleRad)); // calculate direction
    double tipLength = .2 + 0.4 * (angle % 180) / 360;
    int lineType = 8;
    int thickness = 2;
    cv::arrowedLine(frame, center, center + direction, color, thickness, lineType, 0, tipLength); // draw arrow!
}

unsigned long long time_in_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long t_stamp_ms = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
    return t_stamp_ms;
}

void addRoadUserfromTracker(const std::vector<Tracker> &trackers, Message *m, geodetic_converter::GeodeticConverter &gc, const cv::Mat& maskOrient, double *adfGeoTransform, cv::Mat H)
{
    m->t_stamp_ms = time_in_ms();
    m->objects.clear();
    double lat, lon, alt;

    for (auto t : trackers)
    {
        if (t.pred_list_.size() > 0)
        {
            Categories cat;
            switch (t.class_)
            {
            case 0:
                cat = Categories::C_person;
                break;
            case 1:
                cat = Categories::C_car;
                break;
            case 2:
                cat = Categories::C_car;
                break;
            case 3:
                cat = Categories::C_bus;
                break;
            case 4:
                cat = Categories::C_motorbike;
                break;
            case 5:
                cat = Categories::C_bycicle;
                break;
            }
            //std::cout << t.pred_list_.size() << std::endl;
            gc.enu2Geodetic(t.pred_list_.back().x_, t.pred_list_.back().y_, 0, &lat, &lon, &alt);

            int pix_x, pix_y;
            coord2pixel(lat, lon, pix_x, pix_y, adfGeoTransform);

            // TODO: test correctness - added perspective transform call to converter pix_x and pix_y
            // sometimes some values are wrong. float ok? 
            // std::vector<cv::Point2f> map_p, camera_p;
            // std::cout<<"--- pix_x, pix_y: "<<pix_x<<", "<<pix_y<<std::endl;
            // map_p.push_back(cv::Point2f(pix_x, pix_y));
            // std::cout<<"map_p: "<<map_p<<std::endl;
            // //transform camera pixel to map pixel
            // cv::perspectiveTransform(map_p, camera_p, H.inv());
            // std::cout<<"size H: "<<H.cols<<", "<<H.rows<<std::endl;
            // std::cout<<"camera_p: "<<camera_p<<std::endl;    
            // // TODO: in some cases these lines causes seg fault!
            // std::cout<<"y, x :"<<camera_p[0].y<<",  "<<camera_p[0].x<<std::endl;
            // std::cout<<"size maskorient: "<<maskOrient.cols<<", "<<maskOrient.rows<<std::endl;
            // // std::cout<<"vec3b: "<<(cv::Vec3b)(pix_y,pix_x);
            // assert (camera_p[0].x < maskOrient.cols);
            // assert (camera_p[0].y < maskOrient.rows);
            // uint8_t maskOrientPixel = maskOrient.at<cv::Vec3b>(camera_p[0].y,camera_p[0].x)[0]; 
            // std::cout<<"boo: "<<maskOrient.at<cv::Vec3b>(camera_p[0].y,camera_p[0].x)<<std::endl;
            // uint8_t orientation;
            // if(maskOrientPixel != 0)
            // {
            //     orientation = maskOrientPixel;
            //     // std::cout<<"orientation given by the mask "<< int(orientation)<<std::endl;
            // }
            // else
            // {
            //     orientation = uint8_t((int((t.pred_list_.back().yaw_ * 57.29 + 360)) % 360) * 17 / 24);
            //     //std::cout<<"orientation given by the tracker "<< int(orientation)<<std::endl;
            // }

            // TODO: to validate -> it works for grayscale image (see demo.cpp, row: "cv::Mat maskOrient = cv::imread(camera->maskFileOrient, 0);")
            // TODO: include perspective transform
            // std::cout<<"y, x :"<<pix_y<<", "<<pix_x<<std::endl;
            // std::cout<<"size maskorient: "<<maskOrient.cols<<", "<<maskOrient.rows<<std::endl;
            // std::cout<<"point: "<<(cv::Point)(pix_y,pix_x);
            // uint8_t maskOrientPixel = maskOrient.at<uchar>(pix_y,pix_x);
            // uint8_t orientation;
            // if(maskOrientPixel != 0)
            // {
            //     orientation = maskOrientPixel;
            //     // std::cout<<"orientation given by the mask "<< int(orientation)<<std::endl;
            // }
            // else
            // {
            //     orientation = uint8_t((int((t.pred_list_.back().yaw_ * 57.29 + 360)) % 360) * 17 / 24);
            //     //std::cout<<"orientation given by the tracker "<< int(orientation)<<std::endl;
            // }

            uint8_t orientation = uint8_t((int((t.pred_list_.back().yaw_ * 57.29 + 360)) % 360) * 17 / 24);
            // std::cout<<"orient: "<<unsigned(orientation)<<std::endl;
            //std::cout << "lat: " << lat << " lon: " << lon << std::endl;
            uint8_t velocity = uint8_t(std::abs(t.pred_list_.back().vel_ * 3.6 / 2));
            // std::cout<<"vel: "<<unsigned(velocity)<<std::endl;
            RoadUser r{static_cast<float>(lat), static_cast<float>(lon), velocity, orientation, cat};
            //std::cout << std::setprecision(10) << r.latitude << " , " << r.longitude << " " << int(r.speed) << " " << int(r.orientation) << " " << r.category << std::endl;
            m->objects.push_back(r);
        }
    }
    m->num_objects = m->objects.size();
}

void prepare_message(Message *m, const std::vector<ObjCoords> &coords, int idx)
{
    m->cam_idx = idx;
    m->t_stamp_ms = time_in_ms();
    m->num_objects = coords.size();

    m->objects.clear();
    for (unsigned int i = 0; i < coords.size(); i++)
    {
        Categories cat;

        switch (coords[i].class_)
        {
        case 0:
            cat = Categories::C_person;
            break;
        case 1:
            cat = Categories::C_car;
            break;
        case 2:
            cat = Categories::C_car;
            break;
        case 3:
            cat = Categories::C_bus;
            break;
        case 4:
            cat = Categories::C_motorbike;
            break;
        case 5:
            cat = Categories::C_bycicle;
            break;
        }
        RoadUser r{static_cast<float>(coords[i].lat_), static_cast<float>(coords[i].long_), 0, 1, cat};
        std::cout << std::setprecision(10) << r.latitude << " , " << r.longitude << " " << cat << std::endl;
        m->objects.push_back(r);
    }

    m->lights.clear();
}

#endif /*CLASSUTILS_H*/