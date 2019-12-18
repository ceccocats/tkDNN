#include <iostream>

#include "Layer.h"
#include "kernels.h"
#include <math.h>


namespace tk { namespace dnn {

void DeformConv2d::initCUDNN() {

    checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
    checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                                           net->tensorFormat, net->dataType,
                                           1, output_dim.c, 1, 1) );

        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                    net->tensorFormat, net->dataType, output_dim.n, output_dim.c, output_dim.h, output_dim.w));

}

DeformConv2d::DeformConv2d( Network *net, int out_ch, int deformable_group, int kernelH, int kernelW,
                int strideH, int strideW, int paddingH, int paddingW,
                std::string d_fname_weights, std::string fname_weights, bool batchnorm) : 

    LayerWgs(net, net->getOutputDim().c, out_ch, kernelH, kernelW, 1, 
             d_fname_weights, batchnorm, true){

    this->out_ch = out_ch;
    this->deformableGroup = deformable_group;
    this->kernelH = kernelH;
    this->kernelW = kernelW;
    this->strideH = strideH;
    this->strideW = strideW;
    this->paddingH = paddingH;
    this->paddingW = paddingW;
    
    preconv = new tk::dnn::Conv2d(net, deformable_group * 3 * kernelH * kernelW, kernelH, kernelW,
                strideH, strideW, paddingH, paddingW, fname_weights, false);
    net->num_layers--;
    
    output_dim = preconv->output_dim;
    
    output_dim.c = out_ch;
    initCUDNN();
    //allocate data for infer result
    checkCuda( cudaMalloc(&dstData, output_dim.tot()*sizeof(dnnType)) );
}

DeformConv2d::~DeformConv2d() {

    checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
    checkCuda( cudaFree(dstData) );
}

void Conv2dToChunk(int dim, dnnType* srcData, dnnType* offset, dnnType* mask)
{
    // std::cout<<"9\n";
    // cudaMemcpyFromArray(offset, (const struct cudaArray *)srcData, 0, 2*dim.tot()/3, dim.tot()/3, cudaMemcpyDeviceToHost); 
    // checkCuda(cudaMemcpyFromArray(offset, (const struct cudaArray *)srcData, 0, 0, 2*dim, cudaMemcpyDeviceToDevice)); 
    checkCuda(cudaMemcpy(offset, srcData, 2*dim*sizeof(dnnType), cudaMemcpyDeviceToDevice)); 
    
    // std::cout<<"9a\n";
    cudaDeviceSynchronize();
    // cudaMemcpyFromArray(mask, (const struct cudaArray *)srcData, 2*dim.tot()/3, dim.tot(), dim.tot()/3, cudaMemcpyDeviceToHost); 
    // checkCuda(cudaMemcpyFromArray(mask, (const struct cudaArray *)srcData, 0, 2*dim, dim, cudaMemcpyDeviceToDevice)); 
    checkCuda(cudaMemcpy(mask, srcData + 2*dim, dim*sizeof(dnnType), cudaMemcpyDeviceToDevice)); 
    // std::cout<<"9b\n";
}

dnnType* DeformConv2d::infer(dataDim_t &dim, dnnType* srcData) {
    dnnType *input;
    checkCuda(cudaMalloc(&input, dim.tot()*sizeof(dnnType)));
    checkCuda(cudaMemcpy(input, srcData, dim.tot()*sizeof(dnnType), cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();
    srcData = preconv->infer(dim, srcData);
    dim = preconv->output_dim;

    //split to chank
    dnnType *offset, *mask;
    int dst_dim = dim.tot();
    if (dst_dim % 3 != 0 )
        std::cout<<"take attention\n\n";
    int chunk_dim = dst_dim/3;
    checkCuda(cudaMalloc(&offset, 2*chunk_dim*sizeof(dnnType)));
    checkCuda(cudaMalloc(&mask, chunk_dim*sizeof(dnnType)));
    cudaDeviceSynchronize();
    
    Conv2dToChunk(chunk_dim, srcData, offset, mask);

    // kernel sigmoide
    dnnType *vec;
    vec = new dnnType[chunk_dim];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, mask, chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0; i<chunk_dim; i++){
        // std::cout<<i<<" -- "<<vec[i]<<", ";
        vec[i] = 1.0f / (1.0f + exp(-vec[i]));
        // std::cout<<vec[i]<<std::endl;
    }
       
    cudaDeviceSynchronize();
    cudaMemcpy(mask, vec, chunk_dim*sizeof(dnnType), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(vec);

    // dnnType *tmp;
    // cudaMallocHost(&tmp, chunk_dim*sizeof(dnnType));
    // cudaMemcpy(tmp, mask, chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToHost);
    // std::cout<<"conv2d output before chunking"<<std::endl;
    // for (size_t i = 0; i < chunk_dim; i++)
    // {
    //     std::cout<<i<<" -- "<<tmp[i]<<", ";
    // }
    // std::cout<<std::endl;
    // cudaFreeHost(tmp);
 

    const int height_ones = (preconv->input_dim.h + 2 * this->paddingH - (1 * (this->kernelH - 1) + 1)) / this->strideH + 1;
    const int width_ones = (preconv->input_dim.w + 2 * this->paddingW - (1 * (this->kernelW - 1) + 1)) / this->strideW + 1;
    const int dim_ones = preconv->input_dim.c * this->kernelH * this->kernelW * 1 * height_ones * width_ones;
    
    // kernel ones
    dnnType *ones_d1;
    cudaMallocHost(&ones_d1, (height_ones*width_ones)*sizeof(dnnType));
    float aus1[height_ones*width_ones];
    for(int i=0; i<height_ones*width_ones; i++)
        aus1[i]=1.0f;
    cudaMemcpy(ones_d1, aus1, (height_ones*width_ones)*sizeof(dnnType), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    dnnType *ones_d2;
    cudaMallocHost(&ones_d2, dim_ones*sizeof(dnnType));
    float aus2[dim_ones];
    for(int i=0; i<dim_ones; i++)
        aus2[i]=1.0f;
    cudaMemcpy(ones_d2, aus2, (dim_ones)*sizeof(dnnType), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    dcn_v2_cuda_forward(input, this->data_d,
                         this->bias2_d, ones_d1,
                         offset, mask,
                         dstData, ones_d2,
                         this->kernelH, this->kernelW,
                         this->strideH, this->strideW,
                         this->paddingH, this->paddingW,
                         1, 1,
                         this->deformableGroup, 
                         preconv->input_dim.n, preconv->input_dim.c, preconv->input_dim.h, preconv->input_dim.w,
                         this->output_dim.n, this->output_dim.c, this->output_dim.h, this->output_dim.w,
                         dst_dim);

    cudaFree(offset);
    cudaFree(mask);
    cudaFree(input);
    cudaFreeHost(ones_d1);
    cudaFreeHost(ones_d2);
    

    // dnnType *aus3;
    // cudaMallocHost(&aus3, 256*7*7*sizeof(dnnType));
    // cudaMemcpy(aus3, dstData, (256*7*7)*sizeof(dnnType), cudaMemcpyDeviceToHost);
    // checkCuda(cudaDeviceSynchronize());
    // std::cout<<"OutDim:\n";
    // this->output_dim.print();
    // std::cout<<"\n\n\nprint dstData: \n";
    // for (int i = 0 ; i < 256*7*7; i++){
    //     if(i==294)
    //         std::cout<<"\n\n\n";
    //     std::cout<<aus3[i]<<" ";
    // }
    // std::cout<<"\n";
    // cudaFreeHost(aus3);

    std::cout<<"srcData BN:\n";
    printDeviceVector(64, dstData);

    dnnType alpha = dnnType(1);
    dnnType beta  = dnnType(0);
    if(!batchnorm) {
        // // // bias
        alpha = dnnType(1);
        beta  = dnnType(1);
        checkCUDNN( cudnnAddTensor(net->cudnnHandle,
                                   &alpha, biasTensorDesc, bias_d,
                                   &beta, dstTensorDesc, dstData) );
    } else {
        std::cout<<"LOL\n";
        alpha = dnnType(1);
        beta  = dnnType(0);
        checkCUDNN( cudnnBatchNormalizationForwardInference(net->cudnnHandle,
                                                CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                                                dstTensorDesc, dstData, dstTensorDesc,
                                                dstData, biasTensorDesc, //same tensor descriptor as bias
                                                scales_d, bias_d, mean_d, variance_d,
                                                CUDNN_BN_MIN_EPSILON) );
    }
    //update data dimensions
    std::cout<<"dstData BN:\n";
    printDeviceVector(64, dstData);
    dim = output_dim;
    return dstData;
}


}}
