#include<iostream>
#include "tkDNN/ImuOdom.h"

const char *i0_bin   = "imuodom/layers/input0.bin";
const char *i1_bin   = "imuodom/layers/input1.bin";
const char *i2_bin   = "imuodom/layers/input2.bin";
const char *o0_bin   = "imuodom/layers/output0.bin";
const char *o1_bin   = "imuodom/layers/output1.bin";

int main() {

    // V1 
    downloadWeightsifDoNotExist(i0_bin, "imuodom", "https://cloud.hipert.unimore.it/s/ZAy34K5w2ixED6x/download");
    
    // V2
    //downloadWeightsifDoNotExist(i0_bin, "imuodom", "https://cloud.hipert.unimore.it/s/BBSEbEbQbPKxp4s/download");
    
    tk::dnn::ImuOdom ImuNet;
    ImuNet.init("imuodom/layers/");

    const int N = 19513;

    // Network layout
    tk::dnn::dataDim_t dim0(1, 4, 1, 100);
    tk::dnn::dataDim_t dim1(1, 3, 1, 100);
    tk::dnn::dataDim_t dim2(1, 3, 1, 100);

    // Load input
    dnnType *i0_d, *i1_d, *i2_d;
    dnnType *i0_h, *i1_h, *i2_h;
    readBinaryFile(i0_bin, dim0.tot()*N, &i0_h, &i0_d);
    readBinaryFile(i1_bin, dim1.tot()*N, &i1_h, &i1_d);
    readBinaryFile(i2_bin, dim2.tot()*N, &i2_h, &i2_d);

    dnnType *data;
    tk::dnn::dataDim_t dim;

    dnnType *out0, *out1;
    dnnType *out0_h, *out1_h;
    readBinaryFile(o0_bin, ImuNet.odim0.tot()*N, &out0_h, &out0);
    readBinaryFile(o1_bin, ImuNet.odim1.tot()*N, &out1_h, &out1);

    
    std::ofstream path("path.txt");

    int ret_cudnn = 0; 
    for(int i=0; i<N; i++) {
        std::cout<<"i: "<<i<<"\n";
        //TKDNN_TSTART
        // Inference
        ImuNet.update(i0_h, i1_h, i2_h);
        //TKDNN_TSTOP

        // log path
        path<<ImuNet.odomPOS(0)<<" "<<ImuNet.odomPOS(1)<<" "<< ImuNet.odomPOS(2)<<" ";
        path<<ImuNet.odomEULER(0)<<" "<<ImuNet.odomEULER(1)<<" "<< ImuNet.odomEULER(2)<<"\n";

        path.flush();

        // Print real test
        printCenteredTitle( (std::string(" CHECK RESULT ") + std::to_string(i) + " ").c_str() , '=');
        ImuNet.odim0.print();
        ret_cudnn |= checkResult(ImuNet.odim0.tot(), out0, ImuNet.o0_d) == 0 ? 0 : ERROR_CUDNN;
        ImuNet.odim1.print();
        ret_cudnn |= checkResult(ImuNet.odim0.tot(), out1, ImuNet.o1_d) == 0 ? 0 : ERROR_CUDNN;

        i0_h += ImuNet.dim0.tot();
        i1_h += ImuNet.dim1.tot();
        i2_h += ImuNet.dim2.tot();
        out0 += ImuNet.odim0.tot();
        out1 += ImuNet.odim1.tot();
    }

    int err = 0;
    err = system("cat path.txt  | cut -d\" \" -f1,2 | gnuplot -p -e \"set datafile separator ' '; plot '-'\"");
    err = system("cat path.txt  | cut -d\" \" -f6   | gnuplot -p -e \"set datafile separator ' '; plot '-'\"");
    return ret_cudnn;
}
