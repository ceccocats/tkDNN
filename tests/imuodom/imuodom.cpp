#include<iostream>
#include "tkDNN/ImuOdom.h"

const char *i0_bin   = "../tests/imuodom/layers/input0.bin";
const char *i1_bin   = "../tests/imuodom/layers/input1.bin";
const char *i2_bin   = "../tests/imuodom/layers/input2.bin";
const char *o0_bin   = "../tests/imuodom/layers/output0.bin";
const char *o1_bin   = "../tests/imuodom/layers/output1.bin";

const char *c0_bin = "../tests/imuodom/layers/conv1d_7.bin";
const char *c1_bin = "../tests/imuodom/layers/conv1d_8.bin";
const char *c2_bin = "../tests/imuodom/layers/conv1d_9.bin";
const char *c3_bin = "../tests/imuodom/layers/conv1d_10.bin";
const char *c4_bin = "../tests/imuodom/layers/conv1d_11.bin";
const char *c5_bin = "../tests/imuodom/layers/conv1d_12.bin";
const char *l0_bin = "../tests/imuodom/layers/bidirectional_3.bin";
const char *l1_bin = "../tests/imuodom/layers/bidirectional_4.bin";
const char *d0_bin = "../tests/imuodom/layers/dense_3.bin";
const char *d1_bin = "../tests/imuodom/layers/dense_4.bin";


int main() {

    downloadWeightsifDoNotExist(i0_bin, "../tests/imuodom", "https://cloud.hipert.unimore.it/s/ZAy34K5w2ixED6x/download");
    
    tk::dnn::ImuOdom ImuNet;
    ImuNet.init("../tests/imuodom/layers/");

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

    for(int i=0; i<N; i++) {
        std::cout<<"i: "<<i<<"\n";
        //TIMER_START
        // Inference
        ImuNet.update(i0_h, i1_h, i2_h);
        //TIMER_STOP

        // log path
        path<<ImuNet.odomPOS(0)<<" "<<ImuNet.odomPOS(1)<<" "<< ImuNet.odomPOS(2)<<"\n";
        path.flush();

        // Print real test
        //printCenteredTitle( (std::string(" CHECK RESULT ") + std::to_string(i) + " ").c_str() , '=');
        //ImuNet.odim0.print();
        //checkResult(ImuNet.odim0.tot(), out0, ImuNet.o0_d);
        //ImuNet.odim1.print();
        //checkResult(ImuNet.odim0.tot(), out1, ImuNet.o1_d);

        i0_h += ImuNet.dim0.tot();
        i1_h += ImuNet.dim1.tot();
        i2_h += ImuNet.dim2.tot();
        out0 += ImuNet.odim0.tot();
        out1 += ImuNet.odim1.tot();
    }
    return 0;
}
