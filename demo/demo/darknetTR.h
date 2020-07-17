#ifndef DEMO_H
#define DEMO_H

#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>
#include <malloc.h>
#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"
#include "utils.h"
extern "C"
{
typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct {
    float x, y, w, h;
}BOX;

typedef struct {
    int cl;
    BOX bbox;
    float prob;
    char name[20];

}detection;

tk::dnn::Yolo3Detection* load_network(char* net_cfg, int n_classes, int n_batch);
}
#endif /* DETECTIONNN_H*/