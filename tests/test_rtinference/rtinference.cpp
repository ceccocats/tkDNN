#include<iostream>
#include<algorithm>
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

    int ret_tensorrt = 0; 
    std::cout<<"Testing with batchsize: "<<BATCH_SIZE<<"\n";
    std::vector<double> stats;
    printCenteredTitle(" TENSORRT inference ", '=', 30); 
    float total_time = 0;
    for(int i=0; i<64; i++) {

        // generate input
        for(int j=0; j<netRT.input_dim.tot(); j++) {
            dnnType val = ((float) rand() / (RAND_MAX));
            for(int b=0; b<BATCH_SIZE; b++)
                input[netRT.input_dim.tot()*b + j] = val;
        }
        checkCuda(cudaMemcpy(input_d, input, idim.tot()*sizeof(dnnType), cudaMemcpyHostToDevice));

        tk::dnn::dataDim_t dim = idim;
        TKDNN_TSTART
        netRT.infer(dim, input_d);
        TKDNN_TSTOP
        total_time+= t_ns;
        if(i> 1)
        stats.push_back(t_ns);

        // control output
        // std::cout<<"Output Buffers: "<<netRT.getBuffersN()-1<<"\n";
        std::cout<<"Img: "<<i<<"\n";
	for(int o=1; o<netRT.getBuffersN(); o++) {
            for(int b=1; b<BATCH_SIZE; b++) {
                dnnType *out_d = (dnnType*) netRT.buffersRT[o];
                dnnType *out0_d = out_d;
                dnnType *outI_d = out_d + netRT.buffersDIM[o].tot()*b;
                ret_tensorrt |= checkResult(netRT.buffersDIM[o].tot(), outI_d, out0_d,true, 10, false) == 0 ? 0 : ERROR_TENSORRT;
            }
        }
    }

    double min = *std::min_element(stats.begin(), stats.end())/BATCH_SIZE;
    double max = *std::max_element(stats.begin(), stats.end())/BATCH_SIZE;
    double mean =0;
    for(int i=0; i<stats.size(); i++) mean += stats[i]; mean /= stats.size();
    mean /=BATCH_SIZE;

    std::cout<<"Min: "<<min<<" ms\n";    
    std::cout<<"Max: "<<max<<" ms\n";    
    std::cout<<"Avg: "<<mean<<" ms\t"<<1000/(mean)<<" FPS\n"<<COL_END;   


    std::ofstream times;
    times.open("times_rtinference.csv", std::ios_base::app);

    std::string net_name;
    removePathAndExtension(argv[1], net_name);

    times << net_name<< "_" << BATCH_SIZE << ";" << mean << ";" << min << ";" << max << ";" << 1000./mean << "\n";

    times.close();

    return ret_tensorrt;
}

