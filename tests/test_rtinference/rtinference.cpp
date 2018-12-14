#include<iostream>
#include "tkdnn.h"
#include <stdlib.h>     /* srand, rand */

int main(int argc, char *argv[]) {

    if(argc < 2 || !fileExist(argv[1]))
        FatalError("unable to read serialRT file");

    //always same test 
    srand (0); 

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(NULL, argv[1]);

    dnnType *input = new float[netRT.input_dim.tot()];
    dnnType *output = new float[netRT.input_dim.tot()];

    printCenteredTitle(" TENSORRT inference ", '=', 30); 
    for(int i=0; i<100; i++) {
        for(int j=0; j<netRT.input_dim.tot(); j++)
            input[j] = ((float) rand() / (RAND_MAX));
        TIMER_START
        checkCuda(  cudaMemcpyAsync(netRT.buffersRT[netRT.buf_input_idx], input, 
                    netRT.input_dim.tot()*sizeof(float), cudaMemcpyHostToDevice, netRT.stream));
        netRT.enqueue();
        checkCuda(  cudaMemcpyAsync(output, netRT.buffersRT[netRT.buf_output_idx], 
                                    netRT.output_dim.tot()*sizeof(float), cudaMemcpyDeviceToHost, netRT.stream));
        cudaStreamSynchronize(netRT.stream);
        TIMER_STOP
    }
    
    return 0;
}
