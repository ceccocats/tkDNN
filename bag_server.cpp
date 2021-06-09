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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>

#include<sys/socket.h>	//socket
#include<sys/types.h>
#include<netinet/in.h>

using namespace std;
using namespace cv;

#define PORT 8080

#define FRAME_WIDTH         640
#define FRAME_HEIGHT        480

void error(const char *msg)
{
  perror(msg);
  exit(1);
}   int sockfd, newsockfd, portno, n, imgSize, bytes=0, IM_HEIGHT, IM_WIDTH;;
  socklen_t clilen;
  char buffer[256];
 // struct sockaddr_in serv_addr, cli_addr;
//  sockfd=socket(AF_INET, SOCK_STREAM, 0);
    cv::Mat img;
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

    void init_bag(){tk::dnn::readmAPParams(config_filename, classes,  map_points, map_levels, map_step,
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
    void infr(cv::Mat image){

    cv::Mat frame = image;
    cv::imwrite("/home/baggageai/files/build/test.png", frame);
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
    //try{
    //json::value response;
    //vector<json::value> jsonArray;
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
	
	/*json::value detection;
        detection["label"] = json::value::number(b.cl);
        detection["x"] = json::value::number(b.x);
        detection["y"] = json::value::number(b.y);
        detection["w"] = json::value::number(b.w);
        detection["h"] = json::value::number(b.h);
        detection["prob"] = json::value::number(b.prob);
        jsonArray.push_back(detection);*/
        std::cout<< d.cl << " "<< d.prob << " "<< b.x << " "<< b.y << " "<< b.w << " "<< b.h <<"\n";

        if(show)// draw rectangle for detection
            cv::rectangle(batch_frames[0], cv::Point(d.x, d.y), cv::Point(d.x + d.w, d.y + d.h), cv::Scalar(0, 0, 255), 2);             
    }
    //images.push_back(f);
    
    if(show){
        cv::imshow("detection", batch_frames[0]);
        cv::waitKey(0);
    }
          //  response["detections"] = json::value::array(jsonArray);   //JSON Response

  //  std::cout << timeSinceEpochMillisec() << std::endl;
    return ;
    }

int main()
{
//  int sockfd, newsockfd, portno, n, imgSize, bytes=0, IM_HEIGHT, IM_WIDTH;;
 // socklen_t clilen;
  char buffer[256];
  struct sockaddr_in serv_addr, cli_addr;
  //init_bag()
//  cv::Mat img;

  sockfd=socket(AF_INET, SOCK_STREAM, 0);
  if(sockfd<0) error("ERROR opening socket");

  bzero((char*)&serv_addr, sizeof(serv_addr));
  portno = PORT;

  serv_addr.sin_family=AF_INET;
  serv_addr.sin_addr.s_addr=INADDR_ANY;
  serv_addr.sin_port=htons(portno);

  if(bind(sockfd, (struct sockaddr *) &serv_addr,
          sizeof(serv_addr))<0) error("ERROR on binding");

  listen(sockfd,5);
  clilen=sizeof(cli_addr);

  newsockfd=accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
  if(newsockfd<0) error("ERROR on accept");

  uchar sock[3];
  cout << sock <<endl;
  cout << sock+3 <<endl;

  // bzero(buffer,1024);
  // n = read(newsockfd, buffer, 1023);
  // if(n<0) error("ERROR reading from socket");
  //printf("Here is the message: %s\n", buffer);

  // n=write(newsockfd, "I got your message", 18);
  // if(n<0) error("ERROR writing to socket");
  bool running = true;

  while(running)
  { std::cout<<"t"<<"\n";
    IM_HEIGHT = FRAME_HEIGHT;
    IM_WIDTH = FRAME_WIDTH;
    img = Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);

    imgSize = img.total()*img.elemSize();
    uchar sockData[imgSize];
    std::cout<<"t2"<<"\n";
    for(int i=0;i<imgSize;i+=bytes)
      if ((bytes=recv(newsockfd, sockData+i, imgSize-i,0))==-1) error("recv failed");

    int ptr=0;

    for(int i=0;i<img.rows;++i)
      for(int j=0;j<img.cols;++j)
      {
        img.at<Vec3b>(i,j) = Vec3b(sockData[ptr+0],sockData[ptr+1],sockData[ptr+2]);
        ptr=ptr+3;
      }
    std::cout<<"t3"<<"\n";
    int height = img.cols;
    std::cout<<height;
    infr(img)
//    namedWindow( "Server", CV_WINDOW_AUTOSIZE );// Create a window for display.
  //  imshow( "Server", img );
//    char key = waitKey(30);
  //  running = key;
    //esc
   // if(key==27) running =false;
  }

  close(newsockfd);
  close(sockfd);

  return 0;
}
}
