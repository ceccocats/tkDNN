#ifndef INT8BATCHSTREAM_H
#define INT8BATCHSTREAM_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include <iterator>
#include <stdint.h>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <signal.h>
#include <stdlib.h>
#ifdef __linux__    
#include <unistd.h>
#endif

#include <mutex>

#include "NvInfer.h"
#include "utils.h"
#include "tkdnn.h"

/*
 * BatchStream implements the stream for the INT8 calibrator. 
 * It reads the two files .txt with the list of image file names 
 * and the list of label file names. 
 * It then iterates on images and labels.
 */
class BatchStream {
public:
	BatchStream(tk::dnn::dataDim_t dim, int batchSize, int maxBatches, const std::string& fileimglist, const std::string& filelabellist);
	virtual ~BatchStream() { }
	void reset(int firstBatch);
	bool next();
	void skip(int skipCount);
	float *getBatch() { return mBatch.data(); }
	float *getLabels() { return mLabels.data(); }
	int getBatchesRead() const { return mBatchCount; }
	int getBatchSize() const { return mBatchSize; }
	nvinfer1::DimsNCHW getDims() const { return mDims; }
	float* getFileBatch() { return &mFileBatch[0]; }
	float* getFileLabels() { return &mFileLabels[0]; }
	void readInListFile(const std::string& dataFilePath, std::vector<std::string>& mListIn);
	void readCVimage(std::string inputFileName, std::vector<float>& res, bool fixshape = true);
	void readLabels(std::string inputFileName ,std::vector<float>& ris);
	bool update();

private:
	int mBatchSize{ 0 };
	int mMaxBatches{ 0 };
	int mBatchCount{ 0 };
	int mFileCount{ 0 };
	int mFileBatchPos{ 0 };
	int mImageSize{ 0 };

	nvinfer1::DimsNCHW mDims;
	std::vector<float> mBatch;
	std::vector<float> mLabels;
	std::vector<float> mFileBatch;
	std::vector<float> mFileLabels;

	int mHeight;
	int mWidth;
	std::string mFileImgList;
	std::vector<std::string> mListImg;
	std::string mFileLabelList;
	std::vector<std::string> mListLabel;
};

#endif //INT8BATCHSTREAM