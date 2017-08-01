#include "utils.h"

void readBinaryFile(const char* fname, int size, value_type** data_h, value_type** data_d, int seek)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }

    if(seek != 0) {
        dataFile.seekg(seek*sizeof(value_type), dataFile.cur);
    }

    int size_b = size*sizeof(value_type);
    *data_h = new value_type[size];
    if (!dataFile.read ((char*) *data_h, size_b)) 
    {
        error_s << "Error reading file " << fname; 
        FatalError(error_s.str());
    }
    
    checkCuda( cudaMalloc(data_d, size_b) );
    checkCuda( cudaMemcpy(*data_d, *data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
}

void printDeviceVector(int size, value_type* vec_d)
{
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

int checkResult(int size, value_type *data_d, value_type *correct_d) {

    value_type *data_h, *correct_h;
    data_h = new value_type[size];
    correct_h = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(data_h, data_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(correct_h, correct_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);

    int diffs = 0;
    for(int i=0; i<size; i++) {
        if(fabs(data_h[i] - correct_h[i]) > 0.0001) {
            diffs += 1;
            printf("%d\n", i);
        }
    }

    delete [] data_h;
    delete [] correct_h;

    return diffs;
}

void resize(int size, value_type **data)
{
    if (*data != NULL)
        checkCuda( cudaFree(*data) );
    checkCuda( cudaMalloc(data, size*sizeof(value_type)) );
}

void matrixTranspose(cublasHandle_t handle, value_type* srcData, value_type* dstData, int rows, int cols) {

    value_type *A = srcData, *clone = dstData;
    int m = rows, n= cols;
    checkCuda( cudaMemcpy(clone, A, m*n*sizeof(value_type), cudaMemcpyDeviceToDevice));
   
    float const alpha(1.0);
    float const beta(0.0);
    checkERROR( cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, n, &beta, A, m, clone, m ));
}

void matrixMulAdd(  cublasHandle_t handle, value_type* srcData, value_type* dstData, 
                    value_type* add_vector, int dim, value_type mul) {

    checkCuda( cudaMemcpy(dstData, add_vector, dim*sizeof(value_type), cudaMemcpyDeviceToDevice));
        
    value_type alpha = mul;
    checkERROR( cublasSaxpy(handle, dim, &alpha, srcData, 1, dstData, 1));

}