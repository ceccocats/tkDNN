#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define LOCALHOST "127.0.0.1"
#define PORT 8080
#define FRAME_WIDTH         608
#define FRAME_HEIGHT        608

using namespace std;
using namespace cv;

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int main()
{
  int sockfd, portno, n, imgSize, IM_HEIGHT, IM_WIDTH;
  struct sockaddr_in serv_addr;
  struct hostent *server;
  char buffer[256];
  Mat cameraFeed;

  portno = PORT;
  sockfd = socket(AF_INET, SOCK_STREAM, 0);

  if (sockfd < 0) error("ERROR opening socket");

  server = gethostbyname(LOCALHOST);

  if (server == NULL) {
      fprintf(stderr,"ERROR, no such host\n");
      exit(0);
  }

  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy((char *)server->h_addr,
       (char *)&serv_addr.sin_addr.s_addr,
       server->h_length);
  serv_addr.sin_port = htons(portno);

  if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
      error("ERROR connecting");

//  VideoCapture capture;

//	capture.open(0);

 // while(true)
 // {
    /* store image to matrix && test frame */
//    cv::Mat frame=cv::imread(demo.jpg);
    cameraFeed = cv::imread("demo.jpg", cv::IMREAD_COLOR);
    int height = cameraFeed.rows;
    int width = cameraFeed.cols;
   std::cout<<width;
   // Mat cropped = Mat(cameraFeed, Rect(width/2 - width/7,
//                                       height/2 - height/9,
  //                                     2*width/7, 2*height/7));
//    cameraFeed = frame;

    IM_HEIGHT = FRAME_HEIGHT;
    IM_WIDTH = FRAME_WIDTH;

    resize(cameraFeed, cameraFeed, Size( IM_WIDTH , IM_HEIGHT ));

    imgSize=cameraFeed.total()*cameraFeed.elemSize();
    n = send(sockfd, cameraFeed.data, imgSize, 0);
    std::cout<<n;
    if (n < 0) error("ERROR writing to socket");
  //}

  close(sockfd);

  return 0;
}
