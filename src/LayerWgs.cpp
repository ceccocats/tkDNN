#include <iostream>
#include <string.h>

#include "Layer.h"
#include "kernels.h"

namespace tk { namespace dnn {

LayerWgs::LayerWgs(Network *net, int inputs, int outputs, 
                   int kh, int kw, int kl, 
                   std::string fname_weights, bool batchnorm, bool additional_bias, bool deConv, int groups) : Layer(net) {
    inputs = inputs/groups;
        
    this->inputs  = inputs;
    this->outputs = outputs;    
    this->weights_path  = std::string(fname_weights);

    std::cout<<"Reading weights: I="<<inputs<<" O="<<outputs<<" KERNEL="<<kh<<"x"<<kw<<"x"<<kl<<"\n";
    int seek = 0;
    readBinaryFile(weights_path.c_str(), inputs*outputs*kh*kw*kl, &data_h, &data_d, seek);
    seek += inputs*outputs*kh*kw*kl;
    this->additional_bias = additional_bias;
    if(additional_bias) {
        readBinaryFile(weights_path.c_str(), outputs, &bias2_h, &bias2_d, seek);
        seek += outputs;   
    }
    
    readBinaryFile(weights_path.c_str(), outputs, &bias_h, &bias_d, seek);

    this->batchnorm = batchnorm;
    if(batchnorm) {
        seek += outputs;
        readBinaryFile(weights_path.c_str(), outputs, &scales_h, &scales_d, seek);
        seek += outputs;
        readBinaryFile(weights_path.c_str(), outputs, &mean_h, &mean_d, seek);
        seek += outputs;
        readBinaryFile(weights_path.c_str(), outputs, &variance_h, &variance_d, seek);

        float eps = TKDNN_BN_MIN_EPSILON;

        power_h = new dnnType[outputs];
        for(int i=0; i<outputs; i++) power_h[i] = 1.0f;

        for(int i=0; i<outputs; i++)
            mean_h[i] = mean_h[i] / -sqrt(eps + variance_h[i]); 

        for(int i=0; i<outputs; i++)
            variance_h[i] = 1.0f / sqrt(eps + variance_h[i]);
    }


    if(!net->fp16)
        return;

    //convert to fp16
    int w_size = inputs*outputs*kh*kw*kl;
    data16_h = new __half[w_size];
    cudaMalloc(&data16_d, w_size*sizeof(__half));
    float2half(data_d, data16_d, w_size);
    cudaMemcpy(data16_h, data16_d, w_size*sizeof(__half), cudaMemcpyDeviceToHost);

    if(additional_bias){
        int b2_size = outputs;
        bias216_h = new __half[b2_size];
        cudaMalloc(&bias216_d, w_size*sizeof(__half));
        float2half(bias2_d, bias216_d, b2_size);
        cudaMemcpy(bias216_h, bias216_d, b2_size*sizeof(__half), cudaMemcpyDeviceToHost);
    }

    int b_size = outputs;
    bias16_h = new __half[b_size];
    cudaMalloc(&bias16_d, w_size*sizeof(__half));
    float2half(bias_d, bias16_d, b_size);
    cudaMemcpy(bias16_h, bias16_d, b_size*sizeof(__half), cudaMemcpyDeviceToHost);

    if(batchnorm) {

        power16_h    = new __half[b_size];  
        mean16_h     = new __half[b_size];
        variance16_h = new __half[b_size];
        scales16_h   = new __half[b_size];

        cudaMalloc(&power16_d, b_size*sizeof(__half));
        cudaMalloc(&mean16_d, b_size*sizeof(__half));
        cudaMalloc(&variance16_d, b_size*sizeof(__half));
        cudaMalloc(&scales16_d, b_size*sizeof(__half));

        //temporary buffers
        float *tmp_d;
        cudaMalloc(&tmp_d, b_size*sizeof(float));

        //init power array of ones
        cudaMemcpy(tmp_d, power_h, b_size*sizeof(float), cudaMemcpyHostToDevice);
        float2half(tmp_d, power16_d, b_size);
        cudaMemcpy(power16_h, power16_d, b_size*sizeof(__half), cudaMemcpyDeviceToHost);

        //mean array
        cudaMemcpy(tmp_d, mean_h, b_size*sizeof(float), cudaMemcpyHostToDevice);
        float2half(tmp_d, mean16_d, b_size);
        cudaMemcpy(mean16_h, mean16_d, b_size*sizeof(__half), cudaMemcpyDeviceToHost);

        //convert variance
 
        cudaMemcpy(tmp_d, variance_h, b_size*sizeof(float), cudaMemcpyHostToDevice);
        float2half(tmp_d, variance16_d, b_size);
        cudaMemcpy(variance16_h, variance16_d, b_size*sizeof(__half), cudaMemcpyDeviceToHost);

        //convert scales
        float2half(scales_d, scales16_d, b_size);
        cudaMemcpy(scales16_h, scales16_d, b_size*sizeof(__half), cudaMemcpyDeviceToHost);
    
        cudaFree(tmp_d);
    }
}

LayerWgs::~LayerWgs() {
    releaseHost();
    releaseDevice();
}

}}
