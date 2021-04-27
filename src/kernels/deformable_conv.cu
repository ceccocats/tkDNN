#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>
#include <iostream> 
#include "kernels.h"
#include <errno.h>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 512;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__device__ __host__  float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
  const int height, const int width, float h, float w) {
int h_low = floor(h);
int w_low = floor(w);
int h_high = h_low + 1;
int w_high = w_low + 1;

float lh = h - h_low;
float lw = w - w_low;
float hh = 1 - lh, hw = 1 - lw;

float v1 = ( (h_low >= 0 && w_low >= 0) ? bottom_data[h_low * data_width + w_low]:0);
float v2 = ( (h_low >= 0 && w_high <= width - 1) ? bottom_data[h_low * data_width + w_high]:0);
float v3 = ( (h_high <= height - 1 && w_low >= 0) ? bottom_data[h_high * data_width + w_low]:0);
float v4 = ( (h_high <= height - 1 && w_high <= width - 1) ? bottom_data[h_high * data_width + w_high]:0);

float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
return val;
}

__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
  const float *data_im, const float *data_offset, const float *data_mask,
  const int height, const int width,
  const int batch_size, const int num_channels, const int deformable_group,
  const int height_col, const int width_col,
  float *data_col) {
  CUDA_KERNEL_LOOP(index, n)
  {
    //If n is a power of 2, ( i / n ) is equivalent to ( i ≫ log2 n ) and ( i % n ) is equivalent to ( i & n - 1 ).
    const int ind_on_w = index / width_col;
    const int ind_on_w_on_h = ind_on_w / height_col; 
    const int kk = 3 * 3;
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (ind_on_w) % height_col;
    const int b_col = (ind_on_w_on_h) % batch_size;
    const int c_im = (ind_on_w_on_h) / batch_size;
    const int c_col = c_im * kk;

    // compute deformable group index
    const int deformable_group_index = c_im / (int)(num_channels / deformable_group);

    const int h_in = h_col - 1;
    const int w_in = w_col - 1;
    const int s_col = height_col * width_col;
    const int s_col2 = 2 * s_col;

    const int first_member = w_col + width_col * h_col;
    // float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    float *data_col_ptr = data_col + first_member + s_col * (c_col * batch_size + b_col);
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const int add_ptr = (b_col * deformable_group + deformable_group_index) * kk * s_col;
    const float *data_offset_ptr = data_offset + add_ptr + add_ptr;
    const float *data_mask_ptr = data_mask + add_ptr;

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int iter_member = (i * 3 + j);
        // const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_h_ptr = first_member + s_col2 * iter_member;
        
        // const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = s_col + first_member + s_col2 * iter_member;
        
        // const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = first_member + s_col * iter_member;

        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        const float h_im = offset_h + h_in + i;
        const float w_im = offset_w + w_in + j;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        float val = static_cast<float>(0);
        if (h_im < height && w_im < width && h_im > -1 && w_im > -1) {
          //const float map_h = i * dilation_h + offset_h;
          //const float map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * s_col;
        //data_col_ptr += height_col * width_col;
      }
    }
  }
}

__global__ void modulated_deformable_im2col_gpu_kernel_general_version(const int n,
  const float *data_im, const float *data_offset, const float *data_mask,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_per_deformable_group,
  const int batch_size, const int num_channels, const int deformable_group,
  const int height_col, const int width_col,
  float *data_col) {
  CUDA_KERNEL_LOOP(index, n)
  {
    //If n is a power of 2, ( i / n ) is equivalent to ( i ≫ log2 n ) and ( i % n ) is equivalent to ( i & n - 1 ).
    const int ind_on_w = index / width_col;
    const int ind_on_w_on_h = ind_on_w / height_col; 
    const int kk = kernel_h * kernel_w;
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (ind_on_w) % height_col;
    const int b_col = (ind_on_w_on_h) % batch_size;
    const int c_im = (ind_on_w_on_h) / batch_size;
    const int c_col = c_im * kk;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    const int s_col = height_col * width_col;
    const int s_col2 = 2 * s_col;

    const int first_member = w_col + width_col * h_col;
    // float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    float *data_col_ptr = data_col + first_member + s_col * (c_col * batch_size + b_col);
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const int add_ptr = (b_col * deformable_group + deformable_group_index) * kk * s_col;
    const float *data_offset_ptr = data_offset + add_ptr + add_ptr;

    const float *data_mask_ptr = data_mask + add_ptr;
    
    #pragma unroll
    for (int i = 0; i < kernel_h; ++i) {
      #pragma unroll
      for (int j = 0; j < kernel_w; ++j) {
        const int iter_member = (i * kernel_w + j);
        // const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_h_ptr = first_member + s_col2 * iter_member;
        
        // const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = s_col + first_member + s_col2 * iter_member;
        
        // const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = first_member + s_col * iter_member;

        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        const float h_im = offset_h + h_in + i * dilation_h;
        const float w_im = offset_w + w_in + j * dilation_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        float val = static_cast<float>(0);
        if (h_im < height && w_im < width && h_im > -1 && w_im > -1) {
          //const float map_h = i * dilation_h + offset_h;
          //const float map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * s_col;
        //data_col_ptr += height_col * width_col;
      }
    }
  }
}

void modulatedDeformableIm2colCuda(cudaStream_t stream,
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col,
  const int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  // const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, 
      batch_size, channels, deformable_group, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    FatalError("error in modulatedDeformableIm2colCuda: " + std::string(cudaGetErrorString(err)) + "\n");
}

void modulatedDeformableIm2colCudaGeneralVersion(cudaStream_t stream,
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel_general_version
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kenerl_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    FatalError("error in modulatedDeformableIm2colCudaGeneralVersion: " + std::string(cudaGetErrorString(err)) + "\n");
}

void dcnV2CudaForward(cublasStatus_t stat, cublasHandle_t handle, 
                         float *input, float *weight,
                         float *bias, float *ones,
                         float *offset, float *mask,
                         float *output, float *columns,
                         int kernel_h, int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group, const int batch_id,
                         const int in_n, const int in_c, const int in_h, const int in_w, 
                         const int out_n, const int out_c, const int out_h, const int out_w,
                         const int chunk_dim, cudaStream_t stream)
{  
  // stat and handle have be moved out to preserve 2 - 6 milliseconds every 100. 
  const int batch = batch_id;
  const int channels = in_c;
  const int height = in_h;
  const int width = in_w;

  const int channels_out = out_c;

  const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  
  long m = channels_out;
  long n = height_out * width_out;
  long k = 1;
  float alpha = 1.0;
  float beta = 0.0;
  
  stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
              n, m, k, &alpha, 
              ones, k, bias, k, 
              &beta, output + batch * out_c * out_h * out_w, n);
  if (stat != CUBLAS_STATUS_SUCCESS)
    FatalError("CUBLAS initialization failed\n");

  modulatedDeformableIm2colCuda(stream,
                                input + batch * channels * height * width,  
                                offset,// + b * 2 * int((float)chunk_dim / batch),
                                mask,// + b * int((float)chunk_dim / batch),
                                    1, channels, height, width,
                                    height_out, width_out, deformable_group, columns);
  // modulatedDeformableIm2colCudaGeneralVersion(stream,
  //                                   input, offset,
  //                                   mask,
  //                                   1, channels, height, width,
  //                                   height_out, width_out, kernel_h, kernel_w,
  //                                   pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
  //                                   deformable_group, columns);
  
  //(k * m)  x  (m * n)
  // Y = WC
  k = channels * kernel_h * kernel_w;
  beta = 1.0;
  
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
              n, m, k, &alpha, 
              columns, n, weight, k, 
              &beta, output + batch * out_c * out_h * out_w, n);

  if (stat != CUBLAS_STATUS_SUCCESS)
    FatalError("CUBLAS initialization failed\n");

}
