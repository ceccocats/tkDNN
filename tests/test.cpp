#include<iostream>
#include "Layer.h"

int main() {

    tkDNN::dataDim_t dim(1, 1, 10, 10);
    dim.print();
    tkDNN::Network net;
    tkDNN::Dense d(&net, dim, 2, "ci", "lol");

    return 0;
}