#include "visualization.h"

/* Thread function to show the updated images 
**/
void *show_updates(void *x_void_ptr)
{
    cv::namedWindow("original", cv::WINDOW_NORMAL);
    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    cv::namedWindow("topview", cv::WINDOW_NORMAL);
    cv::namedWindow("disparity", cv::WINDOW_NORMAL);
    cv::Mat original_loc, detection_loc, topview_loc, disparity_loc;
    bool update_o_loc, update_de_loc, update_t_loc, update_di_loc;

    while (gRun)
    {
        TIMER_START
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        if (updates.mutex_o.try_lock())
        {
            update_o_loc = updates.update_o;
            updates.update_o = false;
            if (update_o_loc)
                original_loc = updates.original.clone();
            updates.mutex_o.unlock();
        }

        if (updates.mutex_de.try_lock())
        {
            update_de_loc = updates.update_de;
            updates.update_de = false;
            if (update_de_loc)
                detection_loc = updates.detection.clone();
            updates.mutex_de.unlock();
        }

        if (updates.mutex_t.try_lock())
        {
            update_t_loc = updates.update_t;
            updates.update_t = false;
            if (update_t_loc)
                topview_loc = updates.topview.clone();
            updates.mutex_t.unlock();
        }

        if (updates.mutex_di.try_lock())
        {
            update_di_loc = updates.update_di;
            updates.update_di = false;
            if (update_di_loc)
                disparity_loc = updates.disparity.clone();
            updates.mutex_di.unlock();
        }

        if (update_o_loc)
            cv::imshow("original", original_loc);
        if (update_de_loc)
            cv::imshow("detection", detection_loc);
        if (update_t_loc)
            cv::imshow("topview", topview_loc);
        if (update_di_loc)
            cv::imshow("disparity", disparity_loc);
        cv::waitKey(1);
        // usleep(20000);  //sleep 20 msec
        std::cout << "show_updates: ";
        TIMER_STOP
    }
    return (void *)0;
}

void *originalFrame(void *x_void_ptr)
{
    Frame_t *info_show_orig = (Frame_t *)x_void_ptr;
    cv::Mat frame_loc;
    int frame_nbr_loc = 0;
    while (gRun)
    {
        TIMER_START
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        info_show_orig->sem_vc.lock();
        frame_loc = info_show_orig->frame.clone();
        frame_nbr_loc = info_show_orig->frame_nbr;
        info_show_orig->sem_vc.unlock();
        if (frame_nbr_loc == 0)
        {
            usleep(1000000);
            printf("no frame received\n");
            continue;
        }
        updates.mutex_o.lock();
        updates.original = frame_loc.clone();
        updates.update_o = true;
        updates.mutex_o.unlock();
        usleep(10000); //sleep 10 msec
        std::cout << "originalFrame: ";
        TIMER_STOP
    }
    return (void *)0;
}

void *detectionFrame(void *x_void_ptr)
{
    ModFrame_t *info_show = (ModFrame_t *)x_void_ptr;
    double lat, lon, alt;
    int pix_x, pix_y;
    cv::Mat original_frame_loc;
    std::vector<Tracker> trackers;

    geodetic_converter::GeodeticConverter gc;
    double adfGeoTransform[6];
    cv::Mat H;
    tk::dnn::Yolo3Detection yolo;
    int num_detected;
    cv::Mat mask;

    // box variable
    tk::dnn::box b;
    int x0, w, x1, y0, h, y1;
    int objClass;
    std::string det_class;
    ;
    // float prob;
    cv::Scalar intensity;
    std::vector<cv::Point2f> map_p, camera_p;
    int baseline = 0;
    float fontScale = 0.5;
    int thickness = 2;

    while (gRun)
    {
        TIMER_START
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        info_show->sem.lock();
        original_frame_loc = info_show->original_frame.clone();
        // std::vector<Tracker> trackers;
        trackers = info_show->trackers;
        // geodetic_converter::GeodeticConverter gc;
        gc = info_show->gc;
        for (int i = 0; i < 6; i++)
            adfGeoTransform[i] = info_show->adfGeoTransform[i];
        // cv::Mat H;
        H = info_show->H.clone();
        yolo = info_show->yolo;
        mask = info_show->mask.clone();
        info_show->sem.unlock();

        if (trackers.empty())
        {
            usleep(1000000);
            printf("no data available\n");
            continue;
        }

        num_detected = yolo.detected.size();
        for (int i = 0; i < num_detected; i++)
        {
            b = yolo.detected[i];
            x0 = b.x;
            w = b.w;
            x1 = b.x + w;
            y0 = b.y;
            h = b.h;
            y1 = b.y + h;
            objClass = b.cl;
            det_class = obj_class[b.cl];
            // prob = b.prob;

            intensity = mask.at<uchar>(cv::Point(int(x0 + b.w / 2), y1));

            if (intensity[0] && objClass < 6)
            {
                //std::cout<<objClass<<" ("<<prob<<"): "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
                cv::rectangle(original_frame_loc, cv::Point(x0, y0), cv::Point(x1, y1), yolo.colors[objClass], 2);
                // draw label
                cv::Size textSize = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
                cv::rectangle(original_frame_loc, cv::Point(x0, y0), cv::Point((x0 + textSize.width - 2), (y0 - textSize.height - 2)), yolo.colors[b.cl], -1);
                cv::putText(original_frame_loc, det_class, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255), thickness);
            }
        }

        for (auto t : trackers)
        {
            for (size_t p = 1; p < t.pred_list_.size(); p++)
            {
                gc.enu2Geodetic(t.pred_list_[p].x_, t.pred_list_[p].y_, 0, &lat, &lon, &alt);
                coord2pixel(lat, lon, pix_x, pix_y, adfGeoTransform);

                map_p.clear();
                camera_p.clear();
                map_p.push_back(cv::Point2f(pix_x, pix_y));

                //transform camera pixel to map pixel
                cv::perspectiveTransform(map_p, camera_p, H.inv());
                // std::cout<<"x,y: "<<pix_x<<", "<<pix_y<<std::endl;
                // std::cout<<"map_p: "<<map_p<<std::endl;
                // std::cout<<"camera_p: "<<camera_p<<std::endl;
                // std::cout<<"size original_frame_loc: "<<original_frame_loc.cols<<", "<<original_frame_loc.rows<<std::endl;
                // assert (camera_p[0].x < original_frame_loc.cols);
                // assert (camera_p[0].y < original_frame_loc.rows);
                if (camera_p[0].x < original_frame_loc.cols && camera_p[0].y < original_frame_loc.rows && camera_p[0].x >= 0 && camera_p[0].y >= 0)
                    cv::circle(original_frame_loc, cv::Point(camera_p[0].x, camera_p[0].y), 3.0, cv::Scalar(t.r_, t.g_, t.b_), cv::FILLED, 8, 0);
            }
        }

        updates.mutex_de.lock();
        updates.detection = original_frame_loc.clone();
        updates.update_de = true;
        updates.mutex_de.unlock();

        std::cout << "detectionFrame: ";
        TIMER_STOP
    }
    return (void *)0;
}

void *topviewFrame(void *x_void_ptr)
{
    ModFrame_t *info_show = (ModFrame_t *)x_void_ptr;
    double lat, lon, alt;
    int pix_x, pix_y;
    cv::Mat frame_top;
    cv::Mat original_frame_top;
    // original_frame_top = cv::imread("../demo/demo/data/map/map_geo.jpg");
    original_frame_top = cv::imread("../demo/demo/data/map/MASA_4670.png");
    // original_frame_top = cv::imread("../demo/demo/data/map/MASA_4670_V.png");

    std::vector<Tracker> trackers;
    geodetic_converter::GeodeticConverter gc;
    double adfGeoTransform[6];
    cv::Mat H;
    while (gRun)
    {
        TIMER_START
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        info_show->sem.lock();
        // std::vector<Tracker> trackers;
        trackers = info_show->trackers;
        // geodetic_converter::GeodeticConverter gc;
        gc = info_show->gc;
        for (int i = 0; i < 6; i++)
            adfGeoTransform[i] = info_show->adfGeoTransform[i];
        // cv::Mat H;
        H = info_show->H.clone();
        info_show->sem.unlock();
        if (trackers.empty())
        {
            usleep(1000000);
            printf("no data available\n");
            continue;
        }
        frame_top = original_frame_top.clone();
        for (auto t : trackers)
        {
            for (size_t p = 1; p < t.pred_list_.size(); p++)
            {
                gc.enu2Geodetic(t.pred_list_[p].x_, t.pred_list_[p].y_, 0, &lat, &lon, &alt);
                coord2pixel(lat, lon, pix_x, pix_y, adfGeoTransform);
                if (pix_x < frame_top.cols && pix_y < frame_top.rows && pix_x >= 0 && pix_y >= 0)
                    cv::circle(frame_top, cv::Point(pix_x, pix_y), 7.0, cv::Scalar(t.r_, t.g_, t.b_), cv::FILLED, 8, 0);
            }
        }
        //outputVideo<< frame_top;
        // ------------------------------------------------
        updates.mutex_t.lock();
        updates.topview = frame_top.clone();
        updates.update_t = true;
        updates.mutex_t.unlock();

        std::cout << "topviewFrame: ";
        TIMER_STOP
    }
    return (void *)0;
}

void *disparityFrame(void *x_void_ptr)
{
    Frame_t *info_show_disparity = (Frame_t *)x_void_ptr;
    bool first_iteration = true;
    cv::Mat frame_loc;
    int frame_nbr_loc = 0, pre_frame_nbr_loc = 0;
    auto start_t = std::chrono::steady_clock::now();
    auto step_t = std::chrono::steady_clock::now();
    auto end_t = std::chrono::steady_clock::now();

    // information for the disparity map
    cv::Mat canny, pre_canny, canny_RGB, pre_canny_RGB;
    cv::Mat canny_img;
    cv::Mat disparity_frame;
    while (gRun)
    {
        start_t = std::chrono::steady_clock::now();
        step_t = start_t;
        // critical section: copy the struct in local variable
        // in this way we can unlock the sem for the main thread
        info_show_disparity->sem_vc.lock();
        frame_loc = info_show_disparity->frame.clone();
        frame_nbr_loc = info_show_disparity->frame_nbr;
        info_show_disparity->sem_vc.unlock();
        if (frame_nbr_loc == 0)
        {
            usleep(1000000);
            printf("no frame received\n");
            continue;
        }
        // compute frame disparity only in there is a new frame
        if (frame_nbr_loc - pre_frame_nbr_loc > 0)
        {
            pre_frame_nbr_loc = frame_nbr_loc;
            //preprocessing frame
            step_t = std::chrono::steady_clock::now();
            // src_gray
            //canny_img = img_laplacian(frame_loc, 0);
            cv::Canny(canny_img, canny, 100, 100 * 2);
            // sprintf(buf_frame_crop_name,"../demo/demo/data/img_disparity/%d_%d_canny.jpg",frame_nbr_loc, 999);
            // cv::imwrite(buf_frame_crop_name, canny);
            end_t = std::chrono::steady_clock::now();
            std::cout << " TIME END pre canny : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
            step_t = end_t;
            // std::cout<<"o: "<<frame_loc.cols<<" - "<<frame_loc.rows<<std::endl;
            // std::cout<<"canny: "<<canny.cols<<" - "<<canny.rows<<std::endl;
            // std::cout<<"pre: "<<pre_canny.cols<<" - "<<pre_canny.rows<<std::endl;
            if (!first_iteration)
            {
                // backtorgb = cv::cvtColor(pre_canny,cv::COLOR_GRAY2RGB)
                cv::cvtColor(pre_canny, pre_canny_RGB, cv::COLOR_GRAY2RGB);
                cv::cvtColor(canny, canny_RGB, cv::COLOR_GRAY2RGB);
                //disparity_frame = frame_disparity(pre_canny_RGB, canny_RGB, frame_nbr_loc, 999, 0);
                // std::cout<<"size: "<<disparity_frame.rows<<" - "<<disparity_frame.cols<<std::endl;
                // if (disparity_frame.rows == 0 || disparity_frame.cols == 0)
                //     return -1;
                // if (disparity_frame.empty())
                // { // only fools don't check...
                //     std::cout << "image not loaded !" << std::endl;
                //     return -1;
                // }
                end_t = std::chrono::steady_clock::now();
                std::cout << " TIME canny : frame_disparity : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - step_t).count() << " ms" << std::endl;
                step_t = end_t;

                // //--------------------------------
                // //frame box disparity on the original image
                // step_t_segmentation = std::chrono::steady_clock::now();
                // frame_box_disparity(pre_frame, frame, pre_rois, frame_nbr_loc);
                // // reset pre_rois for the new roi of the current frame
                // // pre_rois.erase(pre_rois.begin(), pre_rois.end());
                // end_t_segmentation = std::chrono::steady_clock::now();
                // std::cout << " TIME Frame disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
                // step_t_segmentation = end_t_segmentation;

                // //frame box disparity on the preprocessed image
                // cv::cvtColor(pre_canny, pre_canny_RGB, cv::COLOR_GRAY2RGB);
                // cv::cvtColor(canny, canny_RGB, cv::COLOR_GRAY2RGB);
                // frame_box_disparity(pre_canny_RGB, canny_RGB, pre_rois, frame_nbr_loc);
                // // reset pre_rois for the new roi of the current frame
                // pre_rois.erase(pre_rois.begin(), pre_rois.end());
                // end_t_segmentation = std::chrono::steady_clock::now();
                // std::cout << " TIME Canny Frame disparity : "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_t_segmentation - step_t_segmentation).count() << " ms"<<std::endl;
                // step_t_segmentation = end_t_segmentation;
                // //---------------------------------

                updates.mutex_di.lock();
                updates.disparity = disparity_frame.clone();
                updates.update_di = true;
                updates.mutex_di.unlock();
            }
            pre_canny = canny.clone();
            if (first_iteration)
                first_iteration = false;
            end_t = std::chrono::steady_clock::now();
            std::cout << "disparityFrame : TIME END pre canny : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count() << " ms" << std::endl;
        }
    }
    return (void *)0;
}
