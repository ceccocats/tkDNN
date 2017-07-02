#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include <cublas_v2.h>
#include <cudnn.h>

#define value_type float


#define TIMER_START timespec start, end;                               \
                    clock_gettime(CLOCK_MONOTONIC, &start);            

#define TIMER_STOP  clock_gettime(CLOCK_MONOTONIC, &end);              \
    double t_ns = ((double)(end.tv_sec - start.tv_sec) * 1.0e9 +       \
                  (double)(end.tv_nsec - start.tv_nsec))/1.0e6;        \
    std::cout<<"Time:"<<std::setw(16)<<t_ns<<" ms\n"; 


/********************************************************
 * Prints the error message, and exits
 * ******************************************************/
#define EXIT_WAIVED 0

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " <<cudnnGetErrorString(status);       \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCuda(status) {                                            \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: "<<cudaGetErrorString(status);          \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkERROR(status) {                                           \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Generic failure: " << status;                         \
      FatalError(_error.str());                                        \
    }                                                                  \
}

void readBinaryFile(const char* fname, int size, value_type** data_h, value_type** data_d);
void printDeviceVector(int size, value_type* vec_d);
void resize(int size, value_type **data);

void matrixTranspose(cublasHandle_t handle, value_type* srcData, value_type* dstData, int rows, int cols);

void matrixMulAdd(  cublasHandle_t handle, value_type* srcData, value_type* dstData, 
                    value_type* add_vector, int dim, value_type mul);
#endif //UTILS_H