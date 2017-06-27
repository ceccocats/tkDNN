#include "utils.h"

void readBinaryFile(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
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

void resize(int size, value_type **data)
{
    if (*data != NULL)
        checkCuda( cudaFree(*data) );
    checkCuda( cudaMalloc(data, size*sizeof(value_type)) );
}