#include <iostream>

#include "Layer.h"
#include "kernels.h"
#include <math.h>


namespace tk { namespace dnn {

void DeformConv2d::initCUDNN() {

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) 
        FatalError("CUBLAS initialization failed\n");
    
    checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
    checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                                           net->tensorFormat, net->dataType,
                                           1, output_dim.c, 1, 1) );

    checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                    net->tensorFormat, net->dataType, output_dim.n, output_dim.c, output_dim.h, output_dim.w));

    const int height_ones = (preconv->input_dim.h + 2 * this->paddingH - (1 * (this->kernelH - 1) + 1)) / this->strideH + 1;
    const int width_ones = (preconv->input_dim.w + 2 * this->paddingW - (1 * (this->kernelW - 1) + 1)) / this->strideW + 1;
    const int dim_ones = preconv->input_dim.c * this->kernelH * this->kernelW * 1 * height_ones * width_ones;

    int dst_dim = preconv->output_dim.tot();
    if( dst_dim % 3 != 0 )
        FatalError("DeformConv2d: the Conv2d output is not divisible by three");
    chunk_dim = dst_dim/3;
    checkCuda( cudaMalloc(&offset, 2*chunk_dim*sizeof(dnnType)));
    checkCuda( cudaMalloc(&mask, chunk_dim*sizeof(dnnType)));
    
    // kernel ones
    checkCuda( cudaMalloc(&ones_d1, (height_ones*width_ones)*sizeof(dnnType)) );
    dnnType *ones_h1;
    checkCuda( cudaMallocHost(&ones_h1, (height_ones*width_ones)*sizeof(dnnType)) );
    for(int i=0; i<height_ones*width_ones; i++)
        ones_h1[i]=1.0f;
    checkCuda( cudaMemcpy(ones_d1, ones_h1, (height_ones*width_ones)*sizeof(dnnType), cudaMemcpyHostToDevice) );
    checkCuda( cudaFreeHost(ones_h1) );
    checkCuda( cudaMalloc(&ones_d2, dim_ones*sizeof(dnnType)) );
    dnnType *ones_h2;
    checkCuda( cudaMallocHost(&ones_h2, dim_ones*sizeof(dnnType)) );
    for(int i=0; i<dim_ones; i++)
        ones_h2[i]=1.0f;
    checkCuda( cudaMemcpy(ones_d2, ones_h2, (dim_ones)*sizeof(dnnType), cudaMemcpyHostToDevice) );
    checkCuda( cudaFreeHost(ones_h2) );
    checkCuda( cudaDeviceSynchronize() );
}

DeformConv2d::DeformConv2d( Network *net, int out_ch, int deformable_group, int kernelH, int kernelW,
                int strideH, int strideW, int paddingH, int paddingW,
                std::string d_fname_weights, std::string fname_weights, bool batchnorm) : 

    LayerWgs(net, net->getOutputDim().c, out_ch, kernelH, kernelW, 1, 
             d_fname_weights, batchnorm, true) {
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
    checkCuda( cudaFree(ones_d1) );
    checkCuda( cudaFree(ones_d2) );
    checkCuda( cudaFree(offset) );
    checkCuda( cudaFree(mask) );
    checkCuda( cudaFree(output_conv) );
    cublasDestroy(handle);
}

dnnType* DeformConv2d::infer(dataDim_t &dim, dnnType* srcData) {

    // conv2d 
    output_conv = preconv->infer(dim, srcData);
    // split conv2d outputs into offset and mask
    checkCuda(cudaMemcpy(offset, output_conv, 2*chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice)); 
    checkCuda(cudaMemcpy(mask, output_conv + 2*chunk_dim, chunk_dim*sizeof(dnnType), cudaMemcpyDeviceToDevice)); 
    // kernel sigmoid
    activationSIGMOIDForward(mask, mask, chunk_dim);

    // deformable convolution
    dcnV2CudaForward(stat, handle, 
                         srcData, this->data_d,
                         this->bias2_d, ones_d1,
                         offset, mask,                      
                         dstData, ones_d2,
                         this->kernelH, this->kernelW,
                         this->strideH, this->strideW,
                         this->paddingH, this->paddingW,
                         1, 1,
                         this->deformableGroup, 0, //batch_id for cudnn is set to 0 (no batch)
                         preconv->input_dim.n, preconv->input_dim.c, preconv->input_dim.h, preconv->input_dim.w,
                         this->output_dim.n, this->output_dim.c, this->output_dim.h, this->output_dim.w,
                         chunk_dim);

    dnnType alpha = dnnType(1);
    dnnType beta  = dnnType(0);
    if(!batchnorm) {
        // bias
        alpha = dnnType(1);
        beta  = dnnType(1);
        checkCUDNN( cudnnAddTensor(net->cudnnHandle,
                                   &alpha, biasTensorDesc, bias_d,
                                   &beta, dstTensorDesc, dstData) );
    } else {
        alpha = dnnType(1);
        beta  = dnnType(0);
        checkCUDNN( cudnnBatchNormalizationForwardInference(net->cudnnHandle,
                                                CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                                                dstTensorDesc, dstData, dstTensorDesc,
                                                dstData, biasTensorDesc, //same tensor descriptor as bias
                                                scales_d, bias_d, mean_d, variance_d,
                                                TKDNN_BN_MIN_EPSILON) );
    }

    //update data dimensions
    dim = output_dim;
    return dstData;
}


}}
