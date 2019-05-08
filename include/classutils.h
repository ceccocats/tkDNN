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


struct obj_coords
{
    double LAT;
    double LONG;
    float cl;
};

void readTiff(char *filename, double *adfGeoTransform)
{
    GDALDataset *poDataset;
    GDALAllRegister();
    poDataset = (GDALDataset *)GDALOpen(filename, GA_ReadOnly);
    if (poDataset != NULL)
    {
        //int colms = poDataset->GetRasterXSize();
        //int rows = poDataset->GetRasterYSize();
        poDataset->GetGeoTransform(adfGeoTransform);
    }
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

void fillMatrix(cv::Mat &H, float *matrix, bool show = false)
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

void convert_coords(struct obj_coords *coords, int i, int x, int y, int detected_class, cv::Mat H, double *adfGeoTransform, int frame_nbr)
{
    double latitude, longitude;
    std::vector<cv::Point2f> x_y, ll;
    x_y.push_back(cv::Point2f(x, y));

    //transform camera pixel to map pixel
    cv::perspectiveTransform(x_y, ll, H);
    //tranform to map pixel to map gps
    pixel2coord(ll[0].x, ll[0].y, latitude, longitude, adfGeoTransform);
    //printf("lat: %f, long:%f \n", latitude, longitude);
    coords[i].LAT = latitude;
    coords[i].LONG = longitude;
    coords[i].cl = detected_class;

    /*if (detected_class == 0)
    {

        struct timeval tv;
        gettimeofday(&tv, NULL);
        unsigned long long t_stamp_ms = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;

        //printf(out_file, "%d %lld %d %d\n",frame_nbr, t_stamp_ms, int(ll[0].x), int(ll[0].y));
        fprintf(out_file, "%d %lld %f %f\n", frame_nbr, t_stamp_ms, coords[i].LAT, coords[i].LONG);
        //printf( "%d %lld %f %f\n", frame_nbr, t_stamp_ms, coords[i].LAT, coords[i].LONG);
    }*/
}

void read_projection_matrix(cv::Mat &H, int &proj_matrix_read, char *path)
{
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    float *proj_matrix = (float *)malloc(9 * sizeof(float));

    fp = fopen(path, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    int i = 0;
    while ((read = getline(&line, &len, fp)) != -1)
    {
        if (3 == sscanf(line, "%f %f %f", &proj_matrix[i * 3 + 0], &proj_matrix[i * 3 + 1], &proj_matrix[i * 3 + 2]))
        {
            i++;
            proj_matrix_read = 1;
        }
    }
    free(line);
    fclose(fp);
    fillMatrix(H, proj_matrix);



    free(proj_matrix);
}

#endif /*CLASSUTILS_H*/