#include "utils.h"
#include <string.h>

void printCenteredTitle(const char *title, char fill, int dim) {

    int len = strlen(title);
    int first = dim/2 + len/2;
    
    if(len >0)
        std::cout<<"\n";
    std::cout.width(first); std::cout.fill(fill); std::cout<<std::right<<title;
    std::cout.width(dim - first); std::cout<<"\n"; 
    std::cout.fill(' ');
}

bool fileExist(const char *fname) {
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    if(!dataFile)
        return false;
    return true;
}

void downloadWeightsifDoNotExist(const std::string& input_bin, const std::string& test_folder, const std::string& weights_url){
    if(!fileExist(input_bin.c_str())){
        std::string mkdir_cmd = "mkdir " + test_folder; 
        std::string wget_cmd = "curl " + weights_url + " --output " + test_folder + "/weights.zip";
#ifdef __linux__
        std::string unzip_cmd = "unzip " + test_folder + "/weights.zip -d" + test_folder;
        std::string rm_cmd = "rm " + test_folder + "/weights.zip";

#elif _WIN32

        std::string unzip_cmd = "7z x " + test_folder + "/weights.zip -o" + test_folder;
#endif 
        int err = 0;
        err = system(mkdir_cmd.c_str());
        err = system(wget_cmd.c_str());
        err = system(unzip_cmd.c_str());
#ifdef __linux__
        err = system(rm_cmd.c_str());
#endif

    }
}


void readBinaryFile(std::string fname, int size, dnnType** data_h, dnnType** data_d, int seek)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }

    if(seek != 0) {
        dataFile.seekg(seek*sizeof(dnnType), dataFile.cur);
    }

    int size_b = size*sizeof(dnnType);
    *data_h = new dnnType[size];
    if (!dataFile.read ((char*) *data_h, size_b)) 
    {
        error_s << "Error reading file " << fname << " with n of float: "<<size;
        error_s << " seek: "<<seek << " size: "<<size_b<<"\n";
        FatalError(error_s.str());
    }
    
    checkCuda( cudaMalloc(data_d, size_b) );
    checkCuda( cudaMemcpy(*data_d, *data_h, size_b, cudaMemcpyHostToDevice) );
}


void printDeviceVector(int size, dnnType* vec_d, bool device){
    dnnType *vec;
    if(device) {
        vec = new dnnType[size];
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy(vec, vec_d, size*sizeof(dnnType), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());
    } else {
        vec = vec_d;
    }
    
    for (int i = 0; i < size; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    if(device)
        delete [] vec;
}

int checkResult(int size, dnnType *data_d, dnnType *correct_d, bool device, int limit) {

    dnnType *data_h, *correct_h;
    const float eps = 0.02f;

    if(device) {
        data_h = new dnnType[size];
        correct_h = new dnnType[size];
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy(data_h, data_d, size*sizeof(dnnType), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(correct_h, correct_d, size*sizeof(dnnType), cudaMemcpyDeviceToHost));
        checkCuda(cudaDeviceSynchronize());

    } else {
        data_h = data_d;
        correct_h = correct_d;
    }
    int diffs = 0;
    for(int i=0; i<size; i++) {
        if(data_h[i] != data_h[i] || correct_h[i] != correct_h[i] || //nan control
           fabs(data_h[i] - correct_h[i]) > eps) {
            diffs += 1;
            if(diffs == 1)
                std::cout<<"\n";
            if(diffs < limit)
                std::cout<<" | [ "<<i<<" ]: "<<data_h[i]<<" "<<correct_h[i]<<"\n";
        }
    }

    if(device) {
        delete [] data_h;
        delete [] correct_h;
    }

    std::cout<<" | ";
    if(diffs == 0)
        std::cout<<COL_GREENB<<"OK";
    else
        std::cout<<COL_REDB<<"Wrongs: "<<diffs;

    std::cout<<COL_END<<" ~"<<eps<<"\n";
    return diffs;
}

float getColor(const int c, const int x, const int max){
    float _colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * _colors[i % 6][c % 3] + ratio*_colors[j % 6][c % 3];
    return r;
}


void resize(int size, dnnType **data){
    if (*data != NULL)
        checkCuda( cudaFree(*data) );
    checkCuda( cudaMalloc(data, size*sizeof(dnnType)) );
}

void matrixTranspose(cublasHandle_t handle, dnnType* srcData, dnnType* dstData, int rows, int cols) {

    dnnType *A = srcData, *clone = dstData;
    int m = rows, n= cols;
    checkCuda( cudaMemcpy(clone, A, m*n*sizeof(dnnType), cudaMemcpyDeviceToDevice));
   
    float const alpha(1.0);
    float const beta(0.0);
    checkERROR( cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, n, &beta, A, m, clone, m ));
}

void matrixMulAdd(  cublasHandle_t handle, dnnType* srcData, dnnType* dstData, 
                    dnnType* add_vector, int dim, dnnType mul) {

    checkCuda( cudaMemcpy(dstData, add_vector, dim*sizeof(dnnType), cudaMemcpyDeviceToDevice));
        
    dnnType alpha = mul;
    checkERROR( cublasSaxpy(handle, dim, &alpha, srcData, 1, dstData, 1));

}


void getMemUsage(double& vm_usage_kb, double& resident_set_kb){
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage_kb     = 0.0;
   resident_set_kb = 0.0;

   ifstream stat_stream("/proc/self/stat",ios_base::in);

   //all the stats
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss;

   stat_stream.close();
#ifdef __linux__
   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
#elif _WIN32
 long page_size_kb = 4096/1024;
#endif

   vm_usage_kb     = vsize / 1024.0;  
   resident_set_kb = rss * page_size_kb;
}

void printCudaMemUsage() {
    size_t free, total;
    checkCuda( cudaMemGetInfo(&free, &total) );	 
    std::cout<<"GPU free memory: "<<double(free)/1e6<<" mb.\n";
}

void removePathAndExtension(const std::string &full_string, std::string &name){
    name = full_string;
    std::string tmp_str = full_string;
	std::string slash = "/";
    std::string dot = ".";
    std::size_t current, previous = 0;

    //remove path /path/to/
	current = tmp_str.find(slash);
    if (current != std::string::npos) {
        while (current != std::string::npos) {
            name = tmp_str.substr(previous, current - previous);
            previous = current + 1;
            current = tmp_str.find(slash, previous);
        }
        name = tmp_str.substr(previous, current - previous);
    }
    // remove extension
    current = name.find(dot);
    previous = 0;
    if (current != std::string::npos) 
        name = name.substr(previous, current);
    
    // std::cout<<"full string: "<<full_string<<" name: "<<name<<std::endl;
}