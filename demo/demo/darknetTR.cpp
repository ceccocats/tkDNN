#include "darknetTR.h"

bool gRun;
bool SAVE_RESULT = false;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

extern "C"
{


void copy_image_from_bytes(image im, unsigned char *pdata)
{
//    unsigned char *data = (unsigned char*)pdata;
//    int i, k, j;
    int w = im.w;
    int h = im.h;
    int c = im.c;
//    for (k = 0; k < c; ++k) {
//        for (j = 0; j < h; ++j) {
//            for (i = 0; i < w; ++i) {
//                int dst_index = i + w * j + w * h*k;
//                int src_index = k + c * i + c * w*j;
//                im.data[dst_index] = (float)data[src_index] / 255.;
//            }
//        }
//    }
    memcpy(im.data, pdata, h * w * c);

}

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
    out.data = (float*)xcalloc(h * w * c, sizeof(float));
    return out;
}

tk::dnn::Yolo3Detection* load_network(char* net_cfg, int n_classes, int n_batch)
{
    std::string net;
    net = net_cfg;
    tk::dnn::Yolo3Detection *detNN = new tk::dnn::Yolo3Detection;
    detNN->init(net, n_classes, n_batch);

    return detNN;
}
#include <typeinfo>
void do_inference(tk::dnn::Yolo3Detection *net, image im)
{
    std::vector<cv::Mat> batch_dnn_input;

    cv::Mat frame(im.h, im.w, CV_8UC3, (unsigned char*)im.data);
    batch_dnn_input.push_back(frame);
    net->update(batch_dnn_input, 1);

}


detection* get_network_boxes(tk::dnn::Yolo3Detection *net, float thresh, int batch_num, int *pnum)
{
    std::vector<std::vector<tk::dnn::box>> batchDetected;
    batchDetected = net->get_batch_detected();
    int nboxes =0;
    std::vector<std::string> classesName = net->get_classesName();
    detection* dets = (detection*)xcalloc(batchDetected[batch_num].size(), sizeof(detection));
    for (int i = 0; i < batchDetected[batch_num].size(); ++i)
    {
        if (batchDetected[batch_num][i].prob > thresh)
        {
            dets[nboxes].cl = batchDetected[batch_num][i].cl;
            strcpy(dets[nboxes].name,classesName[dets[nboxes].cl].c_str());
            dets[nboxes].bbox.x = batchDetected[batch_num][i].x;
            dets[nboxes].bbox.y = batchDetected[batch_num][i].y;
            dets[nboxes].bbox.w = batchDetected[batch_num][i].w;
            dets[nboxes].bbox.h = batchDetected[batch_num][i].h;
            dets[nboxes].prob = batchDetected[batch_num][i].prob;
            nboxes += 1;
        }
    }
    if (pnum) *pnum = nboxes;
    return dets;
}
}



