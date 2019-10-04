#include "calibration.h"

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

void fillMatrix(cv::Mat &H, double *matrix, bool show)
{
    double *vals = (double *)H.data;
    for (int i = 0; i < 9; i++)
    {
        vals[i] = matrix[i];
    }
    if (show)
        std::cout << H << "\n";
}

void read_projection_matrix(cv::Mat &H, char *path)
{
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    double proj_matrix[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int i = 0;
    fp = fopen(path, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1)
    {
        std::cout << line << std::endl;
        std::stringstream ss(line);
        while (ss >> proj_matrix[i])
            i++;
    }
    fclose(fp);
    fillMatrix(H, proj_matrix);
    free(line);
}

void convert_coords(std::vector<ObjCoords> &coords, int x, int y, int detected_class, cv::Mat H, double *adfGeoTransform)
{
    double latitude, longitude;
    std::vector<cv::Point2f> x_y, ll;
    x_y.push_back(cv::Point2f(x, y));
    //transform camera pixel to map pixel
    cv::perspectiveTransform(x_y, ll, H);
    //tranform to map pixel to map gps
    pixel2coord(ll[0].x, ll[0].y, latitude, longitude, adfGeoTransform);

    ObjCoords coord;
    coord.lat_ = latitude;
    coord.long_ = longitude;
    coord.class_ = detected_class;
    coords.push_back(coord);
}
