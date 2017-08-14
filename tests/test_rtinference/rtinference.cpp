#include<iostream>
#include "tkdnn.h"

int main(int argc, char *argv[]) {

    // Network layout
    tkDNN::dataDim_t dim(1, 3, 608, 608, 1);
    tkDNN::Network net(dim);

    if(argc < 2 || !fileExist(argv[1]))
        FatalError("unable to read serialRT file");

    //convert network to tensorRT
    tkDNN::NetworkRT netRT(&net, argv[1]);

    dnnType *data;
    checkCuda(cudaMalloc(&data, dim.tot()*sizeof(dnnType)));

    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        TIMER_START
        data = netRT.infer(dim, data);
        TIMER_STOP
    }
    
    return 0;
}
