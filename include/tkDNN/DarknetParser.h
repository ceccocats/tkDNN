#pragma once
#include <iostream>
#include "tkdnn.h"

namespace tk { namespace dnn {

    struct darknetFields_t{
        std::string type = "";
        int width = 0;
        int height = 0;
        int channels = 3;
    };

    std::string darknetParseType(const std::string& line){
            size_t start = line.find("[");
            size_t end  = line.find("]");
            if( start == std::string::npos || end == std::string::npos)
                return "";
            start++;
            std::string type = line.substr(start, end-start);
            return type;
    }

    bool darknetParseFields(const std::string& line, darknetFields_t &fields){
        return true;
    }

    tk::dnn::Network *darknetAddNet(darknetFields_t &fields) {
        std::cout<<"Add Net: "<<fields.type<<"\n";
        dataDim_t dim(1, fields.channels, fields.height, fields.width);
        return new tk::dnn::Network(dim);
    }

    void darknetAddLayer(tk::dnn::Network *net, darknetFields_t &fields) {
        if(net == nullptr)
            FatalError("Cant add a layer without a Net\n");

        std::cout<<"Add layer: "<<fields.type<<"\n";
        if(fields.type == "convolutional") {

        } else{
            FatalError("layer not supported: " + fields.type);
        }
    }

    tk::dnn::Network* darknetParser(std::string cfg_file) {

        tk::dnn::Network *net = nullptr;

        std::ifstream if_cfg(cfg_file);
        if(!if_cfg.is_open())
            FatalError("cloud not open cfg file: " + cfg_file);

        darknetFields_t fields; // will be filled with layers fields
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
                // end of filled type
                if(fields.type != "") {
                    if(fields.type == "net")
                        net = darknetAddNet(fields);
                    else
                        darknetAddLayer(net, fields);
                }

                // new type
                //std::cout<<"type: "<<type<<"\n";
                fields = darknetFields_t();
                fields.type = type;
                continue;
            }

            if(darknetParseFields(line, fields)) {
                // already parsed do nothing
            } else {
                FatalError("could not parse line: " + line);
            }
        }

        // end of filled type
        if(fields.type != "") {
            darknetAddLayer(net, fields);
        }

        if(net == nullptr) {
            FatalError("net not found\n");
        }
    }
        
    

}}
