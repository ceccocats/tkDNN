#define STB_IMAGE_IMPLEMENTATION
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#ifdef __linux__
#include <unistd.h>
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include <mutex>
#include "utils.h"
#include "baggageDetect.hpp"
#include "handler.h"
#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <functional>
#include <string>
#include <fstream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Yolo3Detection.h"
//#include "CenternetDetection.h"
//#include "MobilenetDetection.h"
#include "evaluation.h"
#include <chrono>
#include <cstdint>
#include <iostream>
using namespace std;
using namespace cv;
#include <fstream>
#include <iostream>
#include <string>
#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include<sys/socket.h>	//socket
#include<sys/types.h>
#include<netinet/in.h>
image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h * w * c, sizeof(float));
    return out;
}
int check_mistakes = 0;
image load_image_file(unsigned char *image_data, int channels, int antilog, int gray, int width, int height)
{
    int w, h, c;
    unsigned char *data = image_data;
    w = width;
    h = height;
    c = channels;

    if (!image_data) {
        if (check_mistakes) getchar();
        return make_image(10, 10, 3);
        
    }
    if (channels) c = channels;

    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                (im).data[dst_index] = (float)image_data[src_index]/255.;
            }
        }
    }
    //free(data);
    return im;
}
cv::Mat image_to_mat(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c*img.h*img.w + y*img.w + x];
                mat.data[y*step + x*img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}
    char ntype = 'y';
    const char *config_filename = "../demo/config.yaml";
    const char * net = "../demo/yolo4_fp32.rt";
//    const char * img_path = "../demo/demo.jpg";
    char * img_data;
    bool show = false;
    bool verbose;
    int classes, map_points, map_levels;
    float map_step, IoU_thresh, conf_thresh;
    tk::dnn::Yolo3Detection yolo;
 //   tk::dnn::CenternetDetection cnet;
//    tk::dnn::MobilenetDetection mbnet;
    tk::dnn::DetectionNN *detNN;
    int n_classes = classes;
    std::vector<tk::dnn::Frame> images;
    std::vector<tk::dnn::box> detected_bbox;
    tk::dnn::Frame f;
    //read parametersi
    handler::handler(utility::string_t url):m_listener(url)
{
    m_listener.support(methods::POST, bind(&handler::handle_post, this, placeholders::_1));

}

string name_from_path(string path)
{
    return path.substr(path.find_last_of("/\\")+1);
}
    void handler::init_bag(){tk::dnn::readmAPParams(config_filename, classes,  map_points, map_levels, map_step,
                IoU_thresh, conf_thresh, verbose);


    //extract network name from rt path
    std::string net_name;
    removePathAndExtension(net, net_name);
    std::cout<<"Network: "<<net_name<<std::endl;

    //open files (if needed)
    //std::ofstream times, memory, coco_json;  
    int n_classes = classes;
//    float conf_threshold=0.001;
    detNN = &yolo;
    detNN->init(net, n_classes, 1, conf_thresh);

    //read images 
   // std::ifstream all_labels(labels_path);
//    std::cout << timeSinceEpochMillisec() << std::endl;
    std::string l_filename;

    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);
    return;}
//    init_bag();
//    int images_done;
 //   for (images_done=0 ; std::getline(all_labels, l_filename) && images_done < n_images ; ++images_done) {
   //     std::cout <<COL_ORANGEB<< "Images done:\t" << images_done<< "\n"<<COL_END;
    void handler::handle_post(http_request request){
  //  init_bag();

//    const char * img_path=
    //tk::dnn::Frame f;
    BOOST_LOG_TRIVIAL(info) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << request.to_string();

    map<utility::string_t, utility::string_t> http_get_vars = uri::split_query(request.request_uri().query());
    map<utility::string_t, utility::string_t>::iterator it = http_get_vars.find("name");
//    std::cout<<request<<"\n";
    //If 'name' is not in the query.
    int len;
    if(it == http_get_vars.end())
    {
        BOOST_LOG_TRIVIAL(error) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << "Image name not passed in query.";
        request.reply(status_codes::UnprocessableEntity,"Please pass image name in the query.");
        return;
    }https://github.com/baggageai/baggageai-code-one.git
    std::cout<<http_get_vars["name"]<<"\n";
    string image_name = (string)http_get_vars["name"];
    string ustring;
    //reading binary data and storing it in a pointer
//    std::vector<std::string> img_data=request.extract_vector();
    request.extract_vector().then([image_name, &ustring, &len](vector<unsigned char> v) {
                ustring = {v.begin(),v.end()};
                len = ustring.size();
            }).wait();
    //printSize(ustring);
    BOOST_LOG_TRIVIAL(info) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << "Detection Started";
    unsigned char * sockData = (unsigned char *)ustring.c_str();
   // std::cout<<ch<<"char"<<len<<"len"<<"\n";
//    cv::Mat frame = cv::imread("demo.jpg", cv::IMREAD_COLOR);

    int channels;
    int h=0,w=0;
    image im;
    unsigned char *data = stbi_load_from_memory(sockData, len, &w, &h, &channels, 0);
    im = load_image_file(data, channels, 0, 0, w, h);
//   float *data = stbi_loadf_from_memory(ch, len, &w, &h, &channels, 0); 
std::cout<<"imagedata"<<h<<"w"<<w<<"ch"<<channels<<"\n";
    string outputPath = "/home/baggageai/files/build/testp.png";
    // Write pictures
   // stbi_write_png(outputPath.c_str(), w, h, channels, data, 0);
    //cv::Mat frame=image_to_mat(data,h,w,channels);
//    cv::Mat frame = cv::Mat(data,h, w, CV_16UC(channels));
 //   cv::imwrite("/home/baggageai/files/build/test.png", img);
 //   cv::Mat gray;
    cv::Mat gray=image_to_mat(im);
    cv::Mat frame;
cv::Mat in[] = {gray, gray,gray};
cv::merge(in, 3, frame);
    cv::imwrite("/home/baggageai/files/build/test1.png", frame);
    std::cout<<frame.channels();
    std::vector<cv::Mat> batch_frames;
    batch_frames.push_back(frame);
    int height = frame.rows;
    int width = frame.cols;
    std::cout<<height<<"width"<<width<<"\n";
//    if(!frame.data) 
 //       break;
    std::vector<cv::Mat> batch_dnn_input;
    batch_dnn_input.push_back(frame.clone());
    std::cout<<"test1"<<"\n";
    //inference 
    detected_bbox.clear();
    detNN->update(batch_dnn_input,1);
    detNN->draw(batch_frames);
    detected_bbox = detNN->detected;
    std::cout<<"test2"<<"\n";
    try{
    json::value response;
    vector<json::value> jsonArray;
    // save detections labels
    for(auto d:detected_bbox){
        //convert detected bb in the same format as label
        //<x_center>/<image_width> <y_center>/<image_width> <width>/<image_width> <height>/<image_width>
        tk::dnn::BoundingBox b;
        b.x = (d.x + d.w/2) / width;
        b.y = (d.y + d.h/2) / height;
        b.w = d.w / width;
        b.h = d.h / height;
        b.prob = d.prob;
        b.cl = d.cl;
        //f.det.push_back(b);
	
	json::value detection;
        detection["label"] = json::value::number(b.cl);
        detection["x"] = json::value::number(b.x);
        detection["y"] = json::value::number(b.y);
        detection["w"] = json::value::number(b.w);
        detection["h"] = json::value::number(b.h);
        detection["prob"] = json::value::number(b.prob);
        jsonArray.push_back(detection);
        std::cout<< d.cl << " "<< d.prob << " "<< b.x << " "<< b.y << " "<< b.w << " "<< b.h <<"\n";

        if(show)// draw rectangle for detection
            cv::rectangle(batch_frames[0], cv::Point(d.x, d.y), cv::Point(d.x + d.w, d.y + d.h), cv::Scalar(0, 0, 255), 2);             
    }
    //images.push_back(f);
    
    if(show){
        cv::imshow("detection", batch_frames[0]);
        cv::waitKey(0);
    }
            response["detections"] = json::value::array(jsonArray);   //JSON Response
        request.reply(status_codes::OK,response.serialize());
//        free(detectboxes);
        BOOST_LOG_TRIVIAL(info) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << "Detection Completed and Response sent";
    }
    catch (exception const& e) {
        BOOST_LOG_TRIVIAL(error) << "[" << name_from_path(string(__FILE__)) << "  " << __LINE__ << "] " << e.what();
        request.reply(status_codes::BadRequest, e.what());
    }
  //  std::cout << timeSinceEpochMillisec() << std::endl;
    return ;
    }

