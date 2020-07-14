#include "Int8BatchStream.h"

#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

BatchStream::BatchStream(tk::dnn::dataDim_t dim, int batchSize, int maxBatches, const std::string& fileimglist, const std::string& filelabellist) {
    mBatchSize = batchSize;
    mMaxBatches = maxBatches;
    mDims = nvinfer1::DimsNCHW{ dim.n, dim.c, dim.h, dim.w };
    mHeight = dim.h;
    mWidth = dim.w;
    mImageSize = mDims.c()*mDims.h()*mDims.w();
    mBatch.resize(mBatchSize*mImageSize, 0);
    mLabels.resize(mBatchSize, 0);
    mFileBatch.resize(mDims.n()*mImageSize, 0);
    mFileLabels.resize(mDims.n(), 0);
    mFileImgList = fileimglist;
    readInListFile(fileimglist, mListImg);
    mFileLabelList = filelabellist;
    readInListFile(filelabellist, mListLabel);

    reset(0);
}

void BatchStream::reset(int firstBatch) {
    mBatchCount = 0;
    mFileCount = 0;
    mFileBatchPos = mDims.n();
    skip(firstBatch);
}

bool BatchStream::next() {
    std::cout<<"Next batch: "<<mBatchCount<<" of "<<mMaxBatches<<"\n";
    if (mBatchCount == mMaxBatches-1)
        return false;

    for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize) {
        assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
        if (mFileBatchPos == mDims.n() && !update())
            return false;

        csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
        std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
        std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
    }
    mBatchCount++;
    return true;
}

void BatchStream::skip(int skipCount) {
    if (mBatchSize >= mDims.n() && mBatchSize%mDims.n() == 0 && mFileBatchPos == mDims.n()) {
        mFileCount += skipCount * mBatchSize / mDims.n();
        return;
    }

    int x = mBatchCount;
    for (int i = 0; i < skipCount; i++)
        next();
    mBatchCount = x;
}

void BatchStream::readInListFile(const std::string& dataFilePath, std::vector<std::string>& mListIn) {
    // dataFilePath contains the list of image paths
    int count = 0;
    FILE* f = fopen(dataFilePath.c_str(), "r");
    if (!f)
        FatalError("failed to open " + dataFilePath);
    
    char str[512];
    while (fgets(str, 512, f) != NULL) {
        for (int i = 0; str[i] != '\0'; ++i) {
            if (str[i] == '\n'){
                str[i] = '\0';
                break;
            }
        }
        count ++;
        mListIn.push_back(str);
        if(count == mMaxBatches)
            break;
    }
    fclose(f);
}

void BatchStream::readCVimage(std::string inputFileName, std::vector<float>& res, bool fixshape) {
    // unaltered original DsImage
    cv::Mat m_OrigImage;
    // letterboxed DsImage given to the network as input
    cv::Mat m_LetterboxImage;
    m_OrigImage = cv::imread(inputFileName, cv::IMREAD_COLOR);

    if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
        FatalError("Unable to open " + inputFileName);

    int m_Height = m_OrigImage.rows;
    int m_Width = m_OrigImage.cols;
    if(fixshape) {
        m_Height = mHeight;
        m_Width = mWidth;
    }
    std::cout<<"image is "<<inputFileName<<": "<<m_Height<<" * "<<m_Width<<std::endl;
    // resize the DsImage with scale
    float dim = std::max(m_Height, m_Width);
    int resizeH = ((m_Height / dim) * m_Height);
    int resizeW = ((m_Width / dim) * m_Width);
    float m_ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(m_Height);

    // Additional checks for images with non even dims
    if ((m_Width - resizeW) % 2) resizeW--;
    if ((m_Height - resizeH) % 2) resizeH--;
    assert((m_Width - resizeW) % 2 == 0);
    assert((m_Height - resizeH) % 2 == 0);

    int m_XOffset = (m_Width - resizeW) / 2;
    int m_YOffset = (m_Height - resizeH) / 2;

    assert(2 * m_XOffset + resizeW == m_Width);
    assert(2 * m_YOffset + resizeH == m_Height);

    // resizing
    cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_CUBIC);
    // letterboxing
    cv::copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset,
                    m_XOffset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    m_LetterboxImage.convertTo(m_LetterboxImage, CV_32FC3, 1 / 255.0);
    // converting to RGB and NCHW format
    m_LetterboxImage = cv::dnn::blobFromImage(m_LetterboxImage);
    res.assign(m_LetterboxImage.begin<float>(), m_LetterboxImage.end<float>());
}

void BatchStream::readLabels(std::string inputFileName, std::vector<float>& ris) {
    std::ifstream is(inputFileName.c_str());
    
    std::string line;
    while (std::getline(is, line))
    {
        std::istringstream iss(line);
        float val;
        if(!(iss >> val)) { break; } // error
        ris.push_back(val);
    }
}

bool BatchStream::update() {
    std::string imgFileName = mListImg[mFileCount];
    std::string labelFileName = mListLabel[mFileCount];
    mFileCount++;

    //read image
    mFileBatch.clear();
    readCVimage(imgFileName, mFileBatch);
    // std::transform(
    //     singleImg_rawData.begin(), singleImg_rawData.end(), mFileBatch.begin(), [](uint8_t val) { return static_cast<float>(val); });
    
    //read label
    mFileLabels.clear();
    readLabels(labelFileName, mFileLabels);
    // std::transform(
    //     singleLabels_rawData.begin(), singleLabels_rawData.end(), mFileLabels.begin(), [](uint8_t val) { return static_cast<float>(val); });
    
    mFileBatchPos = 0;
    return true;
}
