#pragma once
#include <iostream>
#include "tkdnn.h"

namespace tk { namespace dnn {

    struct darknetFields_t{
        int width = 0;
        int height = 0;
        int channels = 0;
        int batch_normalize=0;
        int filters=0;
        int size=0;
        int stride=0;
        int pad=0;
        std::string activation = "";

    };

    std::ostream& operator<<(std::ostream& os, const darknetFields_t& f){
        os << f.width << " " << f.height << " " << f.channels << " " << f.batch_normalize<< " " << f.filters<< " " << f.size<< " " << f.stride << " " << f.pad << " " << f.activation;
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

    bool darknetParseFields(const std::string& line, darknetFields_t& fields){

        std::string name,value;
        if(!divideNameAndValue(line, name, value))
            return false;
        std::cout<<name<<std::endl;
        std::cout<<value<<std::endl;
        if(name == "width")
            fields.width = std::stoi(value);
        else if (name == "height")
            fields.height = std::stoi(value);
        else if (name == "channels")
            fields.channels = std::stoi(value);
        else if (name == "batch_normalize")
            fields.batch_normalize = std::stoi(value);
        else if (name == "filters")
            fields.filters = std::stoi(value);
        
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
