#pragma once
#include <iostream>
#include "tkdnn.h"

namespace tk { namespace dnn {

    tk::dnn::Network* DarknetParser(std::string cfg) {

        tk::dnn::dataDim_t dim;
        tk::dnn::Network *net = new tk::dnn::Network(dim);


    }



}}
