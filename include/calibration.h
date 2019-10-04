#ifndef CALIBRATION_H
#define CALIBRATION_H

#include "gdal.h"
#include <gdal_priv.h>
#include <gdal/gdal.h>
#include "gdal/gdal_priv.h"
#include "gdal/cpl_conv.h"

#include <yaml-cpp/yaml.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <cstring>

struct ObjCoords
{
    double lat_;
    double long_;
    int class_;
};

void readTiff(char *filename, double *adfGeoTransform);
void readCameraCalibrationYaml(const std::string &cameraCalib, cv::Mat &cameraMat, cv::Mat &distCoeff);
void pixel2coord(int x, int y, double &lat, double &lon, double *adfGeoTransform);
void coord2pixel(double lat, double lon, int &x, int &y, double *adfGeoTransform);
void fillMatrix(cv::Mat &H, double *matrix, bool show = false);
void read_projection_matrix(cv::Mat &H, char *path);
void convert_coords(std::vector<ObjCoords> &coords, int x, int y, int detected_class, cv::Mat H, double *adfGeoTransform);

#endif /*CALIBRATION_H*/