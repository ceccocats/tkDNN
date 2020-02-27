#include<iostream>
#include "tkdnn.h"

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

    // Network layout
    tk::dnn::dataDim_t dim0(1, 4, 1, 100);
    tk::dnn::dataDim_t dim1(1, 3, 1, 100);
    tk::dnn::dataDim_t dim2(1, 3, 1, 100);

    // Load input
    dnnType *i0_d, *i1_d, *i2_d;
    dnnType *i0_h, *i1_h, *i2_h;
    readBinaryFile(i0_bin, dim0.tot(), &i0_h, &i0_d);
    readBinaryFile(i1_bin, dim1.tot(), &i1_h, &i1_d);
    readBinaryFile(i2_bin, dim2.tot(), &i2_h, &i2_d);

    tk::dnn::Network   net(dim0);
    tk::dnn::Input     x0  (&net, dim0, i0_d);
    tk::dnn::Conv2d    x0_0(&net, 128, 1, 11, 1, 1, 0, 0, c0_bin);
    tk::dnn::Conv2d    x0_1(&net, 128, 1, 11, 1, 1, 0, 0, c1_bin);
    tk::dnn::Pooling   x0_2(&net, 1, 3, 1, 3, tk::dnn::tkdnnPoolingMode_t::POOLING_MAX);

    tk::dnn::Input     x1  (&net, dim1, i1_d);
    tk::dnn::Conv2d    x1_0(&net, 128, 1, 11, 1, 1, 0, 0, c2_bin);
    tk::dnn::Conv2d    x1_1(&net, 128, 1, 11, 1, 1, 0, 0, c3_bin);
    tk::dnn::Pooling   x1_2(&net, 1, 3, 1, 3, tk::dnn::tkdnnPoolingMode_t::POOLING_MAX);

    tk::dnn::Input     x2  (&net, dim2, i2_d);
    tk::dnn::Conv2d    x2_0(&net, 128, 1, 11, 1, 1, 0, 0, c4_bin);
    tk::dnn::Conv2d    x2_1(&net, 128, 1, 11, 1, 1, 0, 0, c5_bin);
    tk::dnn::Pooling   x2_2(&net, 1, 3, 1, 3, tk::dnn::tkdnnPoolingMode_t::POOLING_MAX);

    tk::dnn::Layer *concat_l[3] = { &x0_2, &x1_2, &x2_2 };
    tk::dnn::Route concat (&net, concat_l, 3);

    tk::dnn::LSTM lstm0(&net, 128, true, l0_bin);
    tk::dnn::LSTM lstm1(&net, 128, false, l1_bin);

    tk::dnn::Dense d0 (&net, 3, d0_bin);

    tk::dnn::Layer *lstm1_l[1] = { &lstm1 };
    tk::dnn::Route lstm1_link (&net, lstm1_l, 1);
    tk::dnn::Dense d1 (&net, 4, d1_bin);
    net.print();

    dnnType *data;
    tk::dnn::dataDim_t dim;

    TIMER_START
    // Inference
    data = net.infer(dim, data);
    TIMER_STOP

    // Print real test
    std::cout<<"\n==== CHECK RESULT ====\n";
    dnnType *out0, *out1;
    dnnType *out0_h, *out1_h;
    readBinaryFile(o0_bin, d0.output_dim.tot(), &out0_h, &out0);
    readBinaryFile(o1_bin, d1.output_dim.tot(), &out1_h, &out1);
    d0.output_dim.print();
    checkResult(d0.output_dim.tot(), d0.dstData, out0);
    d1.output_dim.print();
    checkResult(d1.output_dim.tot(), d1.dstData, out1);
    return 0;
}
