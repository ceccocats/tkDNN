#include<iostream>
#include "tkdnn.h"

const char *input_bin = "../tests/yolo3_berkeley/layers/input.bin";
const char *c0_bin    = "../tests/yolo3_berkeley/layers/c0.bin";
const char *c1_bin    = "../tests/yolo3_berkeley/layers/c1.bin";
const char *c2_bin    = "../tests/yolo3_berkeley/layers/c2.bin";
const char *c3_bin    = "../tests/yolo3_berkeley/layers/c3.bin";
const char *c5_bin    = "../tests/yolo3_berkeley/layers/c5.bin";
const char *c6_bin    = "../tests/yolo3_berkeley/layers/c6.bin";
const char *c7_bin    = "../tests/yolo3_berkeley/layers/c7.bin";
const char *c9_bin    = "../tests/yolo3_berkeley/layers/c9.bin";
const char *c10_bin   = "../tests/yolo3_berkeley/layers/c10.bin";
const char *c12_bin   = "../tests/yolo3_berkeley/layers/c12.bin";
const char *c13_bin   = "../tests/yolo3_berkeley/layers/c13.bin";
const char *c14_bin   = "../tests/yolo3_berkeley/layers/c14.bin";
const char *c16_bin   = "../tests/yolo3_berkeley/layers/c16.bin";
const char *c17_bin   = "../tests/yolo3_berkeley/layers/c17.bin";
const char *c19_bin   = "../tests/yolo3_berkeley/layers/c19.bin";
const char *c20_bin   = "../tests/yolo3_berkeley/layers/c20.bin";
const char *c22_bin   = "../tests/yolo3_berkeley/layers/c22.bin";
const char *c23_bin   = "../tests/yolo3_berkeley/layers/c23.bin";
const char *c25_bin   = "../tests/yolo3_berkeley/layers/c25.bin";
const char *c26_bin   = "../tests/yolo3_berkeley/layers/c26.bin";
const char *c28_bin   = "../tests/yolo3_berkeley/layers/c28.bin";
const char *c29_bin   = "../tests/yolo3_berkeley/layers/c29.bin";
const char *c31_bin   = "../tests/yolo3_berkeley/layers/c31.bin";
const char *c32_bin   = "../tests/yolo3_berkeley/layers/c32.bin";
const char *c34_bin   = "../tests/yolo3_berkeley/layers/c34.bin";
const char *c35_bin   = "../tests/yolo3_berkeley/layers/c35.bin";
const char *c37_bin   = "../tests/yolo3_berkeley/layers/c37.bin";
const char *c38_bin   = "../tests/yolo3_berkeley/layers/c38.bin";
const char *c39_bin   = "../tests/yolo3_berkeley/layers/c39.bin";
const char *c41_bin   = "../tests/yolo3_berkeley/layers/c41.bin";
const char *c42_bin   = "../tests/yolo3_berkeley/layers/c42.bin";
const char *c44_bin   = "../tests/yolo3_berkeley/layers/c44.bin";
const char *c45_bin   = "../tests/yolo3_berkeley/layers/c45.bin";
const char *c47_bin   = "../tests/yolo3_berkeley/layers/c47.bin";
const char *c48_bin   = "../tests/yolo3_berkeley/layers/c48.bin";
const char *c50_bin   = "../tests/yolo3_berkeley/layers/c50.bin";
const char *c51_bin   = "../tests/yolo3_berkeley/layers/c51.bin";
const char *c53_bin   = "../tests/yolo3_berkeley/layers/c53.bin";
const char *c54_bin   = "../tests/yolo3_berkeley/layers/c54.bin";
const char *c56_bin   = "../tests/yolo3_berkeley/layers/c56.bin";
const char *c57_bin   = "../tests/yolo3_berkeley/layers/c57.bin";
const char *c59_bin   = "../tests/yolo3_berkeley/layers/c59.bin";
const char *c60_bin   = "../tests/yolo3_berkeley/layers/c60.bin";
const char *c62_bin   = "../tests/yolo3_berkeley/layers/c62.bin";
const char *c63_bin   = "../tests/yolo3_berkeley/layers/c63.bin";
const char *c64_bin   = "../tests/yolo3_berkeley/layers/c64.bin";
const char *c66_bin   = "../tests/yolo3_berkeley/layers/c66.bin";
const char *c67_bin   = "../tests/yolo3_berkeley/layers/c67.bin";
const char *c69_bin   = "../tests/yolo3_berkeley/layers/c69.bin";
const char *c70_bin   = "../tests/yolo3_berkeley/layers/c70.bin";
const char *c72_bin   = "../tests/yolo3_berkeley/layers/c72.bin";
const char *c73_bin   = "../tests/yolo3_berkeley/layers/c73.bin";
const char *c75_bin   = "../tests/yolo3_berkeley/layers/c75.bin";
const char *c76_bin   = "../tests/yolo3_berkeley/layers/c76.bin";
const char *c77_bin   = "../tests/yolo3_berkeley/layers/c77.bin";
const char *c78_bin   = "../tests/yolo3_berkeley/layers/c78.bin";
const char *c79_bin   = "../tests/yolo3_berkeley/layers/c79.bin";
const char *c80_bin   = "../tests/yolo3_berkeley/layers/c80.bin";
const char *c81_bin   = "../tests/yolo3_berkeley/layers/c81.bin";
const char *g82_bin  = "../tests/yolo3_berkeley/layers/g82.bin";
const char *c84_bin   = "../tests/yolo3_berkeley/layers/c84.bin";
const char *c87_bin   = "../tests/yolo3_berkeley/layers/c87.bin";
const char *c88_bin   = "../tests/yolo3_berkeley/layers/c88.bin";
const char *c89_bin   = "../tests/yolo3_berkeley/layers/c89.bin";
const char *c90_bin   = "../tests/yolo3_berkeley/layers/c90.bin";
const char *c91_bin   = "../tests/yolo3_berkeley/layers/c91.bin";
const char *c92_bin   = "../tests/yolo3_berkeley/layers/c92.bin";
const char *c93_bin   = "../tests/yolo3_berkeley/layers/c93.bin";
const char *g94_bin  = "../tests/yolo3_berkeley/layers/g94.bin";
const char *c96_bin   = "../tests/yolo3_berkeley/layers/c96.bin";
const char *c99_bin   = "../tests/yolo3_berkeley/layers/c99.bin";
const char *c100_bin  = "../tests/yolo3_berkeley/layers/c100.bin";
const char *c101_bin  = "../tests/yolo3_berkeley/layers/c101.bin";
const char *c102_bin  = "../tests/yolo3_berkeley/layers/c102.bin";
const char *c103_bin  = "../tests/yolo3_berkeley/layers/c103.bin";
const char *c104_bin  = "../tests/yolo3_berkeley/layers/c104.bin";
const char *c105_bin  = "../tests/yolo3_berkeley/layers/c105.bin";
const char *g106_bin  = "../tests/yolo3_berkeley/layers/g106.bin";
const char *output_bins[3] = {
    "../tests/yolo3_berkeley/debug/layer82_out.bin",
    "../tests/yolo3_berkeley/debug/layer94_out.bin",
    "../tests/yolo3_berkeley/debug/layer106_out.bin"
};

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 320, 544, 1);
    tk::dnn::Network net(dim);

    tk::dnn::Conv2d     c0   (&net,  32, 3, 3, 1, 1, 1, 1,  c0_bin, true);
    tk::dnn::Activation a0   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c1   (&net,  64, 3, 3, 2, 2, 1, 1,  c1_bin, true);
    tk::dnn::Activation a1   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c2   (&net,  32, 1, 1, 1, 1, 0, 0,  c2_bin, true);
    tk::dnn::Activation a2   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c3   (&net,  64, 3, 3, 1, 1, 1, 1,  c3_bin, true);
    tk::dnn::Activation a3   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s4   (&net, &a1);
    tk::dnn::Conv2d     c5   (&net, 128, 3, 3, 2, 2, 1, 1,  c5_bin, true);
    tk::dnn::Activation a5   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c6   (&net,  64, 1, 1, 1, 1, 0, 0,  c6_bin, true);
    tk::dnn::Activation a6   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c7   (&net, 128, 3, 3, 1, 1, 1, 1,  c7_bin, true);
    tk::dnn::Activation a7   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s8   (&net, &a5);
    tk::dnn::Conv2d     c9   (&net,  64, 1, 1, 1, 1, 0, 0,  c9_bin, true);
    tk::dnn::Activation a9   (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c10  (&net, 128, 3, 3, 1, 1, 1, 1, c10_bin, true);
    tk::dnn::Activation a10  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s11  (&net, &s8);

    tk::dnn::Conv2d     c12  (&net, 256, 3, 3, 2, 2, 1, 1, c12_bin, true);
    tk::dnn::Activation a12  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c13  (&net, 128, 1, 1, 1, 1, 0, 0, c13_bin, true);
    tk::dnn::Activation a13  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c14  (&net, 256, 3, 3, 1, 1, 1, 1, c14_bin, true);
    tk::dnn::Activation a14  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s15  (&net, &a12);

    tk::dnn::Conv2d     c16  (&net, 128, 1, 1, 1, 1, 0, 0, c16_bin, true);
    tk::dnn::Activation a16  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c17  (&net, 256, 3, 3, 1, 1, 1, 1, c17_bin, true);
    tk::dnn::Activation a17  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s18  (&net, &s15);
    tk::dnn::Conv2d     c19  (&net, 128, 1, 1, 1, 1, 0, 0, c19_bin, true);
    tk::dnn::Activation a19  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c20  (&net, 256, 3, 3, 1, 1, 1, 1, c20_bin, true);
    tk::dnn::Activation a20  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s21  (&net, &s18);
    tk::dnn::Conv2d     c22  (&net, 128, 1, 1, 1, 1, 0, 0, c22_bin, true);
    tk::dnn::Activation a22  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c23  (&net, 256, 3, 3, 1, 1, 1, 1, c23_bin, true);
    tk::dnn::Activation a23  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s24  (&net, &s21);
    tk::dnn::Conv2d     c25  (&net, 128, 1, 1, 1, 1, 0, 0, c25_bin, true);
    tk::dnn::Activation a25  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c26  (&net, 256, 3, 3, 1, 1, 1, 1, c26_bin, true);
    tk::dnn::Activation a26  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s27  (&net, &s24);
    tk::dnn::Conv2d     c28  (&net, 128, 1, 1, 1, 1, 0, 0, c28_bin, true);
    tk::dnn::Activation a28  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c29  (&net, 256, 3, 3, 1, 1, 1, 1, c29_bin, true);
    tk::dnn::Activation a29  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s30  (&net, &s27);
    tk::dnn::Conv2d     c31  (&net, 128, 1, 1, 1, 1, 0, 0, c31_bin, true);
    tk::dnn::Activation a31  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c32  (&net, 256, 3, 3, 1, 1, 1, 1, c32_bin, true);
    tk::dnn::Activation a32  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s33  (&net, &s30);
    tk::dnn::Conv2d     c34  (&net, 128, 1, 1, 1, 1, 0, 0, c34_bin, true);
    tk::dnn::Activation a34  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c35  (&net, 256, 3, 3, 1, 1, 1, 1, c35_bin, true);
    tk::dnn::Activation a35  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s36  (&net, &s33);

    tk::dnn::Conv2d     c37  (&net, 512, 3, 3, 2, 2, 1, 1, c37_bin, true);
    tk::dnn::Activation a37  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c38  (&net, 256, 1, 1, 1, 1, 0, 0, c38_bin, true);
    tk::dnn::Activation a38  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c39  (&net, 512, 3, 3, 1, 1, 1, 1, c39_bin, true);
    tk::dnn::Activation a39  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s40  (&net, &a37);

    tk::dnn::Conv2d     c41  (&net, 256, 1, 1, 1, 1, 0, 0, c41_bin, true);
    tk::dnn::Activation a41  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c42  (&net, 512, 3, 3, 1, 1, 1, 1, c42_bin, true);
    tk::dnn::Activation a42  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s43  (&net, &s40);
    tk::dnn::Conv2d     c44  (&net, 256, 1, 1, 1, 1, 0, 0, c44_bin, true);
    tk::dnn::Activation a44  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c45  (&net, 512, 3, 3, 1, 1, 1, 1, c45_bin, true);
    tk::dnn::Activation a45  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s46  (&net, &s43);
    tk::dnn::Conv2d     c47  (&net, 256, 1, 1, 1, 1, 0, 0, c47_bin, true);
    tk::dnn::Activation a47  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c48  (&net, 512, 3, 3, 1, 1, 1, 1, c48_bin, true);
    tk::dnn::Activation a48  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s49  (&net, &s46);
    tk::dnn::Conv2d     c50  (&net, 256, 1, 1, 1, 1, 0, 0, c50_bin, true);
    tk::dnn::Activation a50  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c51  (&net, 512, 3, 3, 1, 1, 1, 1, c51_bin, true);
    tk::dnn::Activation a51  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s52  (&net, &s49);
    tk::dnn::Conv2d     c53  (&net, 256, 1, 1, 1, 1, 0, 0, c53_bin, true);
    tk::dnn::Activation a53  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c54  (&net, 512, 3, 3, 1, 1, 1, 1, c54_bin, true);
    tk::dnn::Activation a54  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s55  (&net, &s52);
    tk::dnn::Conv2d     c56  (&net, 256, 1, 1, 1, 1, 0, 0, c56_bin, true);
    tk::dnn::Activation a56  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c57  (&net, 512, 3, 3, 1, 1, 1, 1, c57_bin, true);
    tk::dnn::Activation a57  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s58  (&net, &s55);
    tk::dnn::Conv2d     c59  (&net, 256, 1, 1, 1, 1, 0, 0, c59_bin, true);
    tk::dnn::Activation a59  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c60  (&net, 512, 3, 3, 1, 1, 1, 1, c60_bin, true);
    tk::dnn::Activation a60  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s61  (&net, &s58);

    tk::dnn::Conv2d     c62  (&net,1024, 3, 3, 2, 2, 1, 1, c62_bin, true);
    tk::dnn::Activation a62  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c63  (&net, 512, 1, 1, 1, 1, 0, 0, c63_bin, true);
    tk::dnn::Activation a63  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c64  (&net,1024, 3, 3, 1, 1, 1, 1, c64_bin, true);
    tk::dnn::Activation a64  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s65  (&net, &a62);

    tk::dnn::Conv2d     c66  (&net, 512, 1, 1, 1, 1, 0, 0, c66_bin, true);
    tk::dnn::Activation a66  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c67  (&net,1024, 3, 3, 1, 1, 1, 1, c67_bin, true);
    tk::dnn::Activation a67  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s68  (&net, &s65);

    tk::dnn::Conv2d     c69  (&net, 512, 1, 1, 1, 1, 0, 0, c69_bin, true);
    tk::dnn::Activation a69  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c70  (&net,1024, 3, 3, 1, 1, 1, 1, c70_bin, true);
    tk::dnn::Activation a70  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s71  (&net, &s68);

    tk::dnn::Conv2d     c72  (&net, 512, 1, 1, 1, 1, 0, 0, c72_bin, true);
    tk::dnn::Activation a72  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c73  (&net,1024, 3, 3, 1, 1, 1, 1, c73_bin, true);
    tk::dnn::Activation a73  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Shortcut   s74  (&net, &s71);

    tk::dnn::Conv2d     c75  (&net, 512, 1, 1, 1, 1, 0, 0, c75_bin, true);
    tk::dnn::Activation a75  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c76  (&net,1024, 3, 3, 1, 1, 1, 1, c76_bin, true);
    tk::dnn::Activation a76  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c77  (&net, 512, 1, 1, 1, 1, 0, 0, c77_bin, true);
    tk::dnn::Activation a77  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c78  (&net,1024, 3, 3, 1, 1, 1, 1, c78_bin, true);
    tk::dnn::Activation a78  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c79  (&net, 512, 1, 1, 1, 1, 0, 0, c79_bin, true);
    tk::dnn::Activation a79  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c80  (&net,1024, 3, 3, 1, 1, 1, 1, c80_bin, true);
    tk::dnn::Activation a80  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c81  (&net,  45, 1, 1, 1, 1, 0, 0, c81_bin, false);
    tk::dnn::Yolo     yolo0  (&net,  10, 3, g82_bin);

    tk::dnn::Layer *m83_layers[1] = { &a79 };
    tk::dnn::Route      m83  (&net, m83_layers, 1);
    tk::dnn::Conv2d     c84  (&net, 256, 1, 1, 1, 1, 0, 0, c84_bin, true);
    tk::dnn::Activation a84  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Upsample   u85  (&net, 2);

    tk::dnn::Layer *m86_layers[2] = { &u85, &s61 };
    tk::dnn::Route      m86  (&net, m86_layers, 2);
    tk::dnn::Conv2d     c87  (&net, 256, 1, 1, 1, 1, 0, 0, c87_bin, true);
    tk::dnn::Activation a87  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c88  (&net, 512, 3, 3, 1, 1, 1, 1, c88_bin, true);
    tk::dnn::Activation a88  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c89  (&net, 256, 1, 1, 1, 1, 0, 0, c89_bin, true);
    tk::dnn::Activation a89  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c90  (&net, 512, 3, 3, 1, 1, 1, 1, c90_bin, true);
    tk::dnn::Activation a90  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c91  (&net, 256, 1, 1, 1, 1, 0, 0, c91_bin, true);
    tk::dnn::Activation a91  (&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Conv2d     c92  (&net, 512, 3, 3, 1, 1, 1, 1, c92_bin, true);
    tk::dnn::Activation a92  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c93  (&net,  45, 1, 1, 1, 1, 0, 0, c93_bin, false);
    tk::dnn::Yolo     yolo1  (&net,  10, 3, g94_bin);

    tk::dnn::Layer *m95_layers[1] = { &a91 };
    tk::dnn::Route      m95  (&net, m95_layers, 1);
    tk::dnn::Conv2d     c96  (&net, 128, 1, 1, 1, 1, 0, 0, c96_bin, true);
    tk::dnn::Activation a96  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Upsample   u97  (&net, 2);

    tk::dnn::Layer *m98_layers[2] = { &u97, &s36 };
    tk::dnn::Route      m98  (&net, m98_layers, 2);
    tk::dnn::Conv2d     c99  (&net, 128, 1, 1, 1, 1, 0, 0, c99_bin, true);
    tk::dnn::Activation a99  (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c100 (&net, 256, 3, 3, 1, 1, 1, 1, c100_bin, true);
    tk::dnn::Activation a100 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c101 (&net, 128, 1, 1, 1, 1, 0, 0, c101_bin, true);
    tk::dnn::Activation a101 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c102 (&net, 256, 3, 3, 1, 1, 1, 1, c102_bin, true);
    tk::dnn::Activation a102 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c103 (&net, 128, 1, 1, 1, 1, 0, 0, c103_bin, true);
    tk::dnn::Activation a103 (&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Conv2d     c104 (&net, 256, 3, 3, 1, 1, 1, 1, c104_bin, true);
    tk::dnn::Activation a104 (&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d     c105 (&net,  45, 1, 1, 1, 1, 0, 0, c105_bin, false);
    tk::dnn::Yolo      yolo2 (&net,  10, 3, g106_bin);

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);
    
    //print network model
    net.print();

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, "yolo3_berkeley.rt");

    // the network have 3 outputs
    tk::dnn::dataDim_t out_dim[3];
    out_dim[0] = yolo0.output_dim; 
    out_dim[1] = yolo1.output_dim; 
    out_dim[2] = yolo2.output_dim;
    dnnType *cudnn_out[3], *rt_out[3]; 

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TIMER_START
        net.infer(dim1, data);    
        TIMER_STOP
        dim1.print();   
    }
    cudnn_out[0] = yolo0.dstData;
    cudnn_out[1] = yolo1.dstData;
    cudnn_out[2] = yolo2.dstData;

    printCenteredTitle(" compute detections ", '=', 30);
    TIMER_START
    int ndets = 0;
    int classes = yolo0.classes;
    tk::dnn::Yolo::detection *dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);
    yolo0.computeDetections(dets, ndets, net.input_dim.w, net.input_dim.h, 0.5);
    yolo1.computeDetections(dets, ndets, net.input_dim.w, net.input_dim.h, 0.5);
    yolo2.computeDetections(dets, ndets, net.input_dim.w, net.input_dim.h, 0.5);
    tk::dnn::Yolo::mergeDetections(dets, ndets, classes);

    for(int j=0; j<ndets; j++) {
        tk::dnn::Yolo::box b = dets[j].bbox;
        int x0   = (b.x-b.w/2.);
        int x1   = (b.x+b.w/2.);
        int y0   = (b.y-b.h/2.);
        int y1   = (b.y+b.h/2.);

        int cl = 0;
        for(int c = 0; c < classes; ++c){
            float prob = dets[j].prob[c];
            if(prob > 0)
                cl = c;
        }
        std::cout<<cl<<": "<<x0<<" "<<y0<<" "<<x1<<" "<<y1<<"\n";
    }
    TIMER_STOP
    
    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TIMER_START
        netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }
    rt_out[0] = (dnnType*)netRT.buffersRT[1];
    rt_out[1] = (dnnType*)netRT.buffersRT[2];
    rt_out[2] = (dnnType*)netRT.buffersRT[3];

    for(int i=0; i<3; i++) {
        printCenteredTitle((std::string(" YOLO ") + std::to_string(i) + " CHECK RESULTS ").c_str(), '=', 30);
        dnnType *out, *out_h;
        int odim = out_dim[i].tot();
        readBinaryFile(output_bins[i], odim, &out_h, &out);
        std::cout<<"CUDNN vs correct"; checkResult(odim, cudnn_out[i], out);
        std::cout<<"TRT   vs correct"; checkResult(odim, rt_out[i], out);
        std::cout<<"CUDNN vs TRT    "; checkResult(odim, cudnn_out[i], rt_out[i]);
    }
    return 0;
}
