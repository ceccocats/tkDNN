#pragma once
#include <iostream>
#include "tkdnn.h"

namespace tk { namespace dnn {

    struct darknetFields_t{
        std::string type = "";
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
        float scale_xy = 0;
        std::vector<int> layers;
        std::string activation = "";

    };

    std::ostream& operator<<(std::ostream& os, const darknetFields_t& f){
        os << f.width << " " << f.height << " " << f.channels << " " << f.batch_normalize<< " " << f.filters << " "  << " " << f.activation;
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
        //std::cout<<name<<std::endl;
        //std::cout<<value<<std::endl;
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
        else if (name == "activation")
            fields.activation = value;
        
        return true;
    }

    tk::dnn::Network *darknetAddNet(darknetFields_t &fields) {
        std::cout<<"Add Net: "<<fields.type<<"\n";
        dataDim_t dim(1, fields.channels, fields.height, fields.width);
        return new tk::dnn::Network(dim);
    }

    void darknetAddLayer(tk::dnn::Network *net, darknetFields_t &f, std::string wgs_path) {
        if(net == nullptr)
            FatalError("Cant add a layer without a Net\n");

        std::cout<<"Add layer: "<<f.type<<"\n";
        if(f.type == "convolutional") {
            std::string wgs = wgs_path + "/c" + std::to_string(net->num_layers) + ".bin";
            printf("%d (%d,%d) (%d,%d) (%d,%d) %s %d %d\n", f.filters, f.size_x, f.size_y, f.stride_x, f.stride_y, f.padding_x, f.padding_y, wgs.c_str(), f.batch_normalize, f.groups);
            new tk::dnn::Conv2d(net, f.filters, f.size_x, f.size_y, f.stride_x, 
                f.stride_y, f.padding_x, f.padding_y, wgs, f.batch_normalize, false, f.groups);

        } else if(f.type == "shortcut") {
            if(f.layers.size() != 1) FatalError("no layers to shortcut\n");
            int layerIdx = net->num_layers + f.layers[0];
            if(layerIdx < 0 || layerIdx >= net->num_layers) FatalError("impossible to shortcut\n");
            std::cout<<"shortcut to "<<layerIdx<<" "<<net->layers[layerIdx]->getLayerName()<<"\n";
            new tk::dnn::Shortcut(net, net->layers[layerIdx]);

        } else if(f.type == "upsample") {
            new tk::dnn::Upsample(net, f.stride_x);

        } else if(f.type == "route") {
            if(f.layers.size() == 0) FatalError("no layers to Route\n");
            std::vector<tk::dnn::Layer*> layers;
            for(int i=0; i<f.layers.size(); i++) {
                int layerIdx = net->num_layers + f.layers[i];
                if(layerIdx < 0 || layerIdx >= net->num_layers) FatalError("impossible to shortcut\n");
                layers.push_back(net->layers[layerIdx]);
            }
            new tk::dnn::Route(net, layers.data(), layers.size());

        } else if(f.type == "yolo") {
            std::string wgs = wgs_path + "/g" + std::to_string(net->num_layers) + ".bin";
            new tk::dnn::Yolo(net, f.classes, f.num, wgs, f.n_mask, f.scale_xy);

        } else{
            FatalError("layer not supported: " + f.type);
        }
    }

    tk::dnn::Network* darknetParser(std::string cfg_file, std::string wgs_path) {

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
                        darknetAddLayer(net, fields, wgs_path);
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
            darknetAddLayer(net, fields, wgs_path);
        }

        if(net == nullptr) {
            FatalError("net not found\n");
        }
    }
        
    

}}
