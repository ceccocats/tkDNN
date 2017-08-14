#include<iostream>
#include "tkdnn.h"

int main(int argc, char *argv[]) {

    if(argc < 2 || !fileExist(argv[1]))
        FatalError("unable to read serialRT file");

    //convert network to tensorRT
    tkDNN::NetworkRT netRT(NULL, argv[1]);

    tkDNN::dataDim_t dim = netRT.input_dim;
    dnnType *data;
    checkCuda(cudaMalloc(&data, dim.tot()*sizeof(dnnType)));

    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim.print();
        TIMER_START
        data = netRT.infer(dim, data);
        TIMER_STOP
        dim.print();
    }
    
    return 0;
}
