#include "kernels.h"


__global__ void interleavedToPlanarKernel(uint8_t *src, float *dst, int s_w, int s_h, int s_c, int d_w, int d_h, float ratio_w, float ratio_h) {

    int x = min( (int)(blockIdx.x * blockDim.x + threadIdx.x), d_w-1);
    int y = min( (int)(blockIdx.y * blockDim.y + threadIdx.y), d_h-1);

    float sum_r=0, sum_g=0, sum_b=0;

    float x_src = (float) x * ratio_w; // + ratio_w/2;
    float y_src = (float) y * ratio_h; // + ratio_h/2;

    int r = (int) y_src;
    int c = (int) x_src;
    float dr = y_src - r;
    float dc = x_src - c;

    sum_r = (float) src[(r * s_w + c) * s_c] * (1.0f - dr) * (1.0f - dc) +
            (float) src[((r + 1) * s_w + c) * s_c] * (dr) * (1.0f - dc) +
            (float) src[(r * s_w + c + 1) * s_c] * (1.0f - dr) * (dc) +
            (float) src[((r + 1) * s_w + c + 1) * s_c] * (dr) * (dc);
    sum_g = (float) src[(r * s_w + c) * s_c + 1] * (1.0f - dr) * (1.0f - dc) +
            (float) src[((r + 1) * s_w + c) * s_c + 1] * (dr) * (1.0f - dc) +
            (float) src[(r * s_w + c + 1) * s_c + 1] * (1.0f - dr) * (dc) +
            (float) src[((r + 1) * s_w + c + 1) * s_c + 1] * (dr) * (dc);
    sum_b = (float) src[(r * s_w + c) * s_c + 2] * (1.0f - dr) * (1.0f - dc) +
            (float) src[((r + 1) * s_w + c) * s_c + 2] * (dr) * (1.0f - dc) +
            (float) src[(r * s_w + c + 1) * s_c + 2] * (1.0f - dr) * (dc) +
            (float) src[((r + 1) * s_w + c + 1) * s_c + 2] * (dr) * (dc);

    dst[y * d_w + x] 					= sum_r;
    dst[y * d_w + x + d_w * d_h] 		= sum_g;
    dst[y * d_w + x + d_w * d_h * 2] 	= sum_b;
}

__global__ void interleavedRGBToPlanarBGRKernel(uint8_t *src, float *dst, int s_w, int s_h, int s_c, int d_w, int d_h, float ratio_w, float ratio_h) {

    int x = min( (int)(blockIdx.x * blockDim.x + threadIdx.x), d_w-1);
    int y = min( (int)(blockIdx.y * blockDim.y + threadIdx.y), d_h-1);

    float sum_r=0, sum_g=0, sum_b=0;

    float x_src = (float) x * ratio_w; // + ratio_w/2;
    float y_src = (float) y * ratio_h; // + ratio_h/2;

    int r = (int) y_src;
    int c = (int) x_src;
    float dr = y_src - r;
    float dc = x_src - c;

    sum_r = (float) src[(r * s_w + c) * s_c] * (1.0f - dr) * (1.0f - dc) +
            (float) src[((r + 1) * s_w + c) * s_c] * (dr) * (1.0f - dc) +
            (float) src[(r * s_w + c + 1) * s_c] * (1.0f - dr) * (dc) +
            (float) src[((r + 1) * s_w + c + 1) * s_c] * (dr) * (dc);
    sum_g = (float) src[(r * s_w + c) * s_c + 1] * (1.0f - dr) * (1.0f - dc) +
            (float) src[((r + 1) * s_w + c) * s_c + 1] * (dr) * (1.0f - dc) +
            (float) src[(r * s_w + c + 1) * s_c + 1] * (1.0f - dr) * (dc) +
            (float) src[((r + 1) * s_w + c + 1) * s_c + 1] * (dr) * (dc);
    sum_b = (float) src[(r * s_w + c) * s_c + 2] * (1.0f - dr) * (1.0f - dc) +
            (float) src[((r + 1) * s_w + c) * s_c + 2] * (dr) * (1.0f - dc) +
            (float) src[(r * s_w + c + 1) * s_c + 2] * (1.0f - dr) * (dc) +
            (float) src[((r + 1) * s_w + c + 1) * s_c + 2] * (dr) * (dc);

    dst[y * d_w + x] 					= sum_b;
    dst[y * d_w + x + d_w * d_h] 		= sum_g;
    dst[y * d_w + x + d_w * d_h * 2] 	= sum_r;
}

void interleavedToPlanar( uint8_t *d_src, float *d_dst, int s_w, int s_h, int s_c, int d_w, int d_h){
	dim3 dg( ceil( (double)d_w/32 ), ceil( (double)d_h/8 ) );
	dim3 db( 32, 8);

	interleavedToPlanarKernel<<< dg, db >>>(d_src, d_dst, s_w, s_h, s_c, d_w, d_h, (float)s_w/d_w, (float)s_h/d_h);
	cudaDeviceSynchronize();
}

void interleavedRGBToPlanarBGR( uint8_t *d_src, float *d_dst, int s_w, int s_h, int s_c, int d_w, int d_h){
	dim3 dg( ceil( (double)d_w/32 ), ceil( (double)d_h/8 ) );
	dim3 db( 32, 8);

	interleavedRGBToPlanarBGRKernel<<< dg, db >>>(d_src, d_dst, s_w, s_h, s_c, d_w, d_h, (float)s_w/d_w, (float)s_h/d_h);
	cudaDeviceSynchronize();
}
