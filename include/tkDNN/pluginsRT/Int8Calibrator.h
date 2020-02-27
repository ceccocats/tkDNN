#include <vector>
#include <assert.h>
#include <algorithm>
#include <iterator>

#include "NvInfer.h"

class BatchStream
{
public:
	BatchStream(tk::dnn::dataDim_t dim, int batchSize, int maxBatches)
	{
        mBatchSize = batchSize;
        mMaxBatches = maxBatches;
		mDims = nvinfer1::DimsNCHW{ dim.n, dim.c, dim.h, dim.w };
		mImageSize = mDims.c()*mDims.h()*mDims.w();
		mBatch.resize(mBatchSize*mImageSize, 0);
		mLabels.resize(mBatchSize, 0);
		mFileBatch.resize(mDims.n()*mImageSize, 0);
		mFileLabels.resize(mDims.n(), 0);
		reset(0);
	}

	void reset(int firstBatch)
	{
		mBatchCount = 0;
		mFileCount = 0;
		mFileBatchPos = mDims.n();
		skip(firstBatch);
	}

	bool next()
	{
        std::cout<<"Next batch: "<<mBatchCount<<" of "<<mMaxBatches<<"\n";
		if (mBatchCount == mMaxBatches)
			return false;

		for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
		{
			assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
			if (mFileBatchPos == mDims.n() && !update())
				return false;

			// copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
			csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
			std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
			std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
		}
		mBatchCount++;
		return true;
	}

	void skip(int skipCount)
	{
		if (mBatchSize >= mDims.n() && mBatchSize%mDims.n() == 0 && mFileBatchPos == mDims.n())
		{
			mFileCount += skipCount * mBatchSize / mDims.n();
            std::cout<<mFileCount<<"\n";
			return;
		}

		int x = mBatchCount;
		for (int i = 0; i < skipCount; i++)
			next();
		mBatchCount = x;
	}

	float *getBatch() { return &mBatch[0]; }
	float *getLabels() { return &mLabels[0]; }
	int getBatchesRead() const { return mBatchCount; }
	int getBatchSize() const { return mBatchSize; }
	nvinfer1::DimsNCHW getDims() const { return mDims; }
private:
	float* getFileBatch() { return &mFileBatch[0]; }
	float* getFileLabels() { return &mFileLabels[0]; }

	bool update()
	{
		std::string inputFileName = std::string("calibBatches/batch") + std::to_string(mFileCount++);
		FILE * file = fopen(inputFileName.c_str(), "rb");
		if (!file) {
            FatalError("cant open batch calib file: " + inputFileName);
        	return false;
        }

		size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.n()*mImageSize, file);
		size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.n(), file);;
		assert(readInputCount == size_t(mDims.n()*mImageSize) && readLabelCount == size_t(mDims.n()));

		fclose(file);
		mFileBatchPos = 0;
		return true;
	}

	int mBatchSize{ 0 };
	int mMaxBatches{ 0 };
	int mBatchCount{ 0 };

	int mFileCount{ 0 }, mFileBatchPos{ 0 };
	int mImageSize{ 0 };

	nvinfer1::DimsNCHW mDims;
	std::vector<float> mBatch;
	std::vector<float> mLabels;
	std::vector<float> mFileBatch;
	std::vector<float> mFileLabels;
};



class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
	Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true)
		: mStream(stream), mReadCache(readCache)
	{
		DimsNCHW dims = mStream.getDims();
		mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();
		checkCuda(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
		mStream.reset(firstBatch);
	}

	virtual ~Int8EntropyCalibrator()
	{
		checkCuda(cudaFree(mDeviceInput));
	}

	int getBatchSize() const override { return mStream.getBatchSize(); }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
        std::cout<<"CALIB request batch\n";
		if (!mStream.next())
			return false;

		checkCuda(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
		bindings[0] = mDeviceInput;
		return true;
	}

	const void* readCalibrationCache(size_t& length) override
	{
		mCalibrationCache.clear();
		std::ifstream input("table.calib", std::ios::binary);
		input >> std::noskipws;
		
        FatalError("rewrite different");
        //if (mReadCache && input.good())
		//	std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output("table.calib", std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

private:
	BatchStream mStream;
	bool mReadCache{ true };

	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache;
};
