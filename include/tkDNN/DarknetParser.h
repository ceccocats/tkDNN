#pragma once
#include <iostream>
#include "tkdnn.h"

namespace tk { namespace dnn {

    struct darknetFields_t{
        int width = 0;
        int height = 0;
        int channels = 0;
        int batch_normalize=0;
        int groups = 0;
        int filters=0;
        int size_x=0;
        int size_y=0;
        int stride_x=0;
        int stride_y=0;
        int padding_x = 0;
        int padding_y = 0;
        int n_mask = 0;
        int classes = 0;
        int num = 0;
        int pad = 0;
        float scale_xy = 0;
        std::vector<int> layers;
        std::string activation = "";

    };

    std::ostream& operator<<(std::ostream& os, const darknetFields_t& f){
        os << f.width << " " << f.height << " " << f.channels << " " << f.batch_normalize<< " " << f.filters << " "  << f.activation<< " " << f.scale_xy;
        return os;
    }

    std::string darknetParseType(const std::string& line){
            size_t start = line.find("[");
            size_t end  = line.find("]");
            if( start == std::string::npos || end == std::string::npos)
                return "";
            start++;
            std::string type = line.substr(start, end-start);
            return type;
    }

    bool divideNameAndValue(const std::string& line, std::string&name, std::string& value){
        size_t sep = line.find("=");
        if(sep == std::string::npos)
            return false;
        
        name   = line.substr(0, sep);   
        value  = line.substr(sep+1, line.size() - (sep+1));   
        return true;
    }

    std::vector<int> fromStringToIntVec(const std::string& line, const char delimiter){
        std::stringstream linestream(line);
        std::string value;
        std::vector<int> values;

        while(getline(linestream,value,delimiter))
            values.push_back(std::stoi(value));
        return values;
    }

    bool darknetParseFields(const std::string& line, darknetFields_t& fields){

        std::string name,value;
        if(!divideNameAndValue(line, name, value))
            return false;
        if(name.find("width") !=  std::string::npos)
            fields.width = std::stoi(value);
        else if(name.find("height") !=  std::string::npos)
            fields.height = std::stoi(value);
        else if(name.find("channels") !=  std::string::npos)
            fields.channels = std::stoi(value);
        else if(name.find("batch_normalize") !=  std::string::npos)
            fields.batch_normalize = std::stoi(value);
        else if(name.find("filters") !=  std::string::npos)
            fields.filters = std::stoi(value);
        else if(name.find("activation") !=  std::string::npos)
            fields.activation = value;
        else if(name.find("size") !=  std::string::npos){
            fields.size_x = std::stoi(value);
            fields.size_y = std::stoi(value);
        }
        else if(name.find("size_x") !=  std::string::npos)
            fields.size_x = std::stoi(value);
        else if(name.find("size_y") !=  std::string::npos)
            fields.size_y = std::stoi(value);
        else if(name.find("stride") !=  std::string::npos){
            fields.stride_x = std::stoi(value);
            fields.stride_y = std::stoi(value);
        }
        else if(name.find("stride_x") !=  std::string::npos)
            fields.stride_x = std::stoi(value);
        else if(name.find("stride_y") !=  std::string::npos)
            fields.stride_y = std::stoi(value);
        else if(name.find("pad") !=  std::string::npos)
            fields.pad = std::stoi(value);
        else if(name.find("classes") !=  std::string::npos)
            fields.classes = std::stoi(value);
        else if(name.find("num") !=  std::string::npos)
            fields.num = std::stoi(value);
        else if(name.find("scale_xy") !=  std::string::npos)
            fields.scale_xy = std::stof(value);
        else if(name.find("from") !=  std::string::npos)
            fields.layers.push_back(std::stof(value));
        else if(name.find("mask") !=  std::string::npos){
            auto vec = fromStringToIntVec(value, ',');
            fields.n_mask = vec.size();
        }
        else if(name.find("layers") !=  std::string::npos)
            fields.layers = fromStringToIntVec(value, ',');

        else
            std::cout<<"Not supported field: "<<line<<std::endl;
        return true;
    }



    tk::dnn::Network* DarknetParser(std::string cfg_file) {

        tk::dnn::dataDim_t dim;
        tk::dnn::Network *net = new tk::dnn::Network(dim);

        std::ifstream if_cfg(cfg_file);
        if(!if_cfg.is_open())
            FatalError("cloud not open cfg file: " + cfg_file);

        std::string line;
        while(std::getline(if_cfg, line)) {
            // remove comments
            std::size_t found = line.find("#");
            if ( found != std::string::npos ) {
                line = line.substr(0, found);
            }

            // skip empty lines
            if(line.size() == 0)
                continue;
            
            std::string type = darknetParseType(line);
            if(type.size() > 0) {
                std::cout<<"type: "<<type<<"\n";
            }
        }
    }
        
    

}}
