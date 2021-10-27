#ifndef YOLO_CONTAINER_H
#define YOLO_CONTAINER_H

namespace tk { namespace dnn {
class YoloRT;
class YoloContainer
{
public:
    YoloRT *yolos[16];
    int n_yolos{};
};

extern YoloContainer yoloContainer;
}}

#endif // YOLO_CONTAINER_H