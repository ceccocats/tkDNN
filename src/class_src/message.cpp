#include "message.h"
#include "calibration.h"

unsigned long long time_in_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long t_stamp_ms = (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
    return t_stamp_ms;
}

/* Convert orientation from radian to quantized degree (from 360 to 255)
**/
uint8_t orientation_to_uint8(float yaw)
{
    // 57.29 is (180/pi) -> conversion in degrees
    // 17 / 24 is (255/360) -> quantization
    // let a yaw in radians, it is converted into degrees (*57.29), 
    // then into positive degrees, then it is quantized into 255. 
    uint8_t orientation = uint8_t((int((yaw * 57.29 + 360)) % 360) * 17 / 24);
    return orientation;
}

/* Convert speed from m/s to quantized km/h every 1/2 km/h
**/
uint8_t speed_to_uint8(float vel)
{
    // 3.6 -> conversion in km/h
    // *2 -> quantization km/h evrey 1/2
    // let a velocity in m/s, it is converted into km/h (*3.6), then (*2) 
    // we achive a double speed. In a urban track we can consider a maximum 
    // speed of 127 km/h. So we can fit 127 on a byte with a multiplication 
    // by 2. Each increment corresponds to a speed greater than 1/2 km/h.
    uint8_t velocity = uint8_t(std::abs(vel * 3.6 * 2));
    return velocity;
}        

void addRoadUserfromTracker(const std::vector<Tracker> &trackers, Message *m, geodetic_converter::GeodeticConverter &gc, const cv::Mat &maskOrient, double *adfGeoTransform, cv::Mat H)
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

            uint8_t orientation = orientation_to_uint8(t.pred_list_.back().yaw_);
            // std::cout<<"orient: "<<unsigned(orientation)<<std::endl;
            //std::cout << "lat: " << lat << " lon: " << lon << std::endl;
            uint8_t velocity = speed_to_uint8(t.pred_list_.back().vel_);
            
            // std::cout<<"vel: "<<unsigned(velocity)<<std::endl;
            RoadUser r{static_cast<float>(lat), static_cast<float>(lon), velocity, orientation, cat};
            //std::cout << std::setprecision(10) << r.latitude << " , " << r.longitude << " " << int(r.speed) << " " << int(r.orientation) << " " << r.category << std::endl;
            m->objects.push_back(r);
        }
    }
    m->num_objects = m->objects.size();
}
