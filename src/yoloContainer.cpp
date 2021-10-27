#include "yoloContainer.h"
#include "pluginsRT/YoloRT.h"

namespace tk { namespace dnn {
YoloContainer yoloContainer;

nvinfer1::IPluginV2* YoloRTCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
  const char * buf = reinterpret_cast<const char*>(serialData),*bufCheck = buf;
  int classes_temp = tk::dnn::readBUF<int>(buf);
  int num_temp = tk::dnn::readBUF<int>(buf);
  int n_masks_temp = tk::dnn::readBUF<int>(buf);
  float scale_xy_temp = tk::dnn::readBUF<float>(buf);
  float nms_thresh_temp = tk::dnn::readBUF<float>(buf);
  int nms_kind_temp = tk::dnn::readBUF<int>(buf);
  int new_coords_temp = tk::dnn::readBUF<int>(buf);

  YoloRT *r = new YoloRT(classes_temp,num_temp,nullptr,n_masks_temp,scale_xy_temp,nms_thresh_temp,nms_kind_temp,new_coords_temp);

  r->c = tk::dnn::readBUF<int>(buf);
  r->h = tk::dnn::readBUF<int>(buf);
  r->w = tk::dnn::readBUF<int>(buf);
  for(int i=0; i<r->n_masks; i++)
      r->mask[i] = tk::dnn::readBUF<dnnType>(buf);
  for(int i=0; i<r->n_masks*2*r->num; i++)
      r->bias[i] = tk::dnn::readBUF<dnnType>(buf);

  // save classes names
  r->classesNames.resize(r->classes);
  for(int i=0; i<r->classes; i++) {
          char tmp[YOLORT_CLASSNAME_W];
    for(int j=0; j<YOLORT_CLASSNAME_W; j++)
      tmp[j] = tk::dnn::readBUF<char>(buf);
          r->classesNames[i] = std::string(tmp);
  }
  assert(buf == bufCheck + serialLength);

  yoloContainer.yolos[yoloContainer.n_yolos++] = r;
  return r;
}

}}