#include "pluginsRT/RegionRT.h"
#include "pluginsRT/RouteRT.h"
#include "pluginsRT/ReshapeRT.h"
#include "pluginsRT/FlattenConcatRT.h"
#include "pluginsRT/YoloRT.h"

// Static class fields initialization
namespace tk { namespace dnn {
nvinfer1::PluginFieldCollection RegionRTCreator::mFC{};
std::vector<nvinfer1::PluginField> RegionRTCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(RegionRTCreator);

nvinfer1::PluginFieldCollection RouteRTCreator::mFC{};
std::vector<nvinfer1::PluginField> RouteRTCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(RouteRTCreator);

nvinfer1::PluginFieldCollection ReshapeRTCreator::mFC{};
std::vector<nvinfer1::PluginField> ReshapeRTCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ReshapeRTCreator);

nvinfer1::PluginFieldCollection FlattenConcatRTCreator::mFC{};
std::vector<nvinfer1::PluginField> FlattenConcatRTCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FlattenConcatRTCreator);

nvinfer1::PluginFieldCollection YoloRTCreator::mFC{};
std::vector<nvinfer1::PluginField> YoloRTCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(YoloRTCreator);
}}