#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "cameraUtils.h"
#include <iostream>
#include <cstring>

#include <yaml-cpp/yaml.h>

struct Parameters_t 
{
    char *net;
    char *tiffile;
    int n_cameras;
    Camera_t *cameras;
};

void readCamerasParametersYaml(const std::string &camerasParams, Parameters_t *par);
bool read_parameters(int argc, char *argv[], Parameters_t *par);

#endif /*CONFIGURATION_H*/