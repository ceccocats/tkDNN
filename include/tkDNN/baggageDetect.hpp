
#ifdef OS_WIN
	#pragma once

	#ifdef LIB_EXPORTS
	#define LIB_API __declspec(dllexport)
	#else
	#define LIB_API __declspec(dllimport)
	#endif
#endif

#include <iostream>


#include <vector>
#include <string>


#ifdef OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
using namespace cv;
#endif


using namespace std;

struct baggagedetector {
	int x,y,w,h,size;
	char *label;
	float prob;
};

#ifdef __cplusplus
class baggageAI
{
	//std::shared_ptr<void> detector_gpu_ptr;
	public:
		//static LIB_API image_t image_load(std::string image_filename);
		#ifdef OS_WIN
			LIB_API baggageAI();
			//LIB_API ~baggageAI();
			LIB_API baggagedetector * baggageDetections(char *input);
			LIB_API baggagedetector * baggageDetections(unsigned char *input, int len, int antiLog, int gray);
			#ifdef OPENCV
				LIB_API baggagedetector * baggageDetections(Mat m);
			#endif
		#else
			baggageAI();
			//LIB_API ~baggageAI();
			baggagedetector * baggageDetections(char *input);
			baggagedetector * baggageDetections(unsigned char *input, int len,int antiLog, int gray);
			#ifdef OPENCV
				baggagedetector * baggageDetections(Mat m);
			#endif
		#endif
};

#endif
