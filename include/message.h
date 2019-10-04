#ifndef MESSAGE_H
#define MESSAGE_H

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
// #include <sys/socket.h> //socket
// #include <arpa/inet.h>  //inet_addr
// #include <unistd.h>     //write

#include "tracker.h"

#include "../masa_protocol/include/send.hpp"
#include "../masa_protocol/include/serialize.hpp"

unsigned long long time_in_ms();

void addRoadUserfromTracker(const std::vector<Tracker> &trackers, Message *m, geodetic_converter::GeodeticConverter &gc, const cv::Mat &maskOrient, double *adfGeoTransform, cv::Mat H);

#endif /*MESSAGE_H*/