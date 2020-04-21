#include<iostream>
#include "tkdnn.h"
#include <stdlib.h>     /* srand, rand */


int main(int argc, char *argv[]) {

    if(argc < 2 || !fileExist(argv[1]))
        FatalError("unable to read serialRT file");

    int BATCH_SIZE = 1;
    if(argc >2)
        BATCH_SIZE = atoi(argv[2]);

    //always same test 
    srand (0); 

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(NULL, argv[1]);
    
    tk::dnn::dataDim_t idim = netRT.input_dim;
    tk::dnn::dataDim_t odim = netRT.output_dim;
    idim.n = BATCH_SIZE;
    odim.n = BATCH_SIZE;
    dnnType *input = new float[idim.tot()];
    dnnType *output = new float[odim.tot()];
    dnnType *input_d;
    checkCuda( cudaMalloc(&input_d, idim.tot()*sizeof(dnnType)));

    std::cout<<"Testing with batchsize: "<<BATCH_SIZE<<"\n";
    printCenteredTitle(" TENSORRT inference ", '=', 30); 
    for(int i=0; i<10; i++) {
        for(int j=0; j<idim.tot(); j++) {
            input[j] = ((float) rand() / (RAND_MAX));
        }
        checkCuda(cudaMemcpy(input_d, input, idim.tot()*sizeof(dnnType), cudaMemcpyHostToDevice));

        tk::dnn::dataDim_t dim = idim;
        TIMER_START
        netRT.infer(dim, input_d);
        TIMER_STOP
    }
    
    return 0;
}
