#pragma once
#include <iostream>
#include "tkDNN/tkdnn.h"

namespace tk { namespace dnn {

    struct darknetFields_t{
        std::string type = "";
        int width = 0;
        int height = 0;
        int channels = 3;
        int batch_normalize=0;
        int groups = 1;
        int group_id = 0;
        int filters=1;
        int size_x=1;
        int size_y=1;
        int stride_x=1;
        int stride_y=1;
        int padding_x = 0;
        int padding_y = 0;
        int n_mask = 0;
        int classes = 20;
        int num = 1;
        int pad = 0;
        int coords = 4;
        int nms_kind = 0;
        int new_coords= 0;
        float scale_xy = 1;
        float nms_thresh = 0.45;
        std::vector<int> layers;
        std::string activation = "linear";

        friend std::ostream& operator<<(std::ostream& os, const darknetFields_t& f){
            os << f.width << " " << f.height << " " << f.channels << " " << f.batch_normalize<< " " << f.filters << " "  << f.activation<< " " << f.scale_xy;
            return os;
        }
    };

    std::string darknetParseType(const std::string& line);
    bool divideNameAndValue(const std::string& line, std::string&name, std::string& value);
    std::vector<int> fromStringToIntVec(const std::string& line, const char delimiter);
    
    bool darknetParseFields(const std::string& line, darknetFields_t& fields);
    tk::dnn::Network *darknetAddNet(darknetFields_t &fields);
    void darknetAddLayer(tk::dnn::Network *net, darknetFields_t &f, std::string wgs_path, 
                         std::vector<tk::dnn::Layer*> &netLayers, const std::vector<std::string>& names);
    std::vector<std::string> darknetReadNames(const std::string& names_file);
    tk::dnn::Network* darknetParser(const std::string& cfg_file, const std::string& wgs_path, const std::string& names_file);

}}
