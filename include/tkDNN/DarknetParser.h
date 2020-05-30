#pragma once
#include <iostream>
#include "tkdnn.h"

namespace tk { namespace dnn {

    tk::dnn::Network* DarknetParser(std::string cfg) {

        tk::dnn::dataDim_t dim;
        tk::dnn::Network *net = new tk::dnn::Network(dim);

    }
        
    std::string parseType(const std::string& line){
            size_t start = line.find("[");
            size_t end  = line.find("]");
            if( start == std::string::npos || end == std::string::npos)
                return "";
            start++;
            std::string type = line.substr(start, end-start);
            return type;
    }

}}
