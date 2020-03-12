#include <iostream>
#include <vector>
#include "tkdnn.h"

int main()
{

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 416, 416, 1);
    tk::dnn::Network net(dim);

    // create csresnext50-panet-spp model
    std::string bin_path = "../tests/csresnext50-panet-spp";
    int classes = 80;
    tk::dnn::Yolo *yolo[3];

    std::string input_bin = bin_path + "/layers/input.bin";
    std::string output_bin = bin_path + "/debug/layer137_out.bin";
    std::vector<std::string> output_bins = {
        bin_path + "/debug/layer115_out.bin",
        bin_path + "/debug/layer126_out.bin",
        bin_path + "/debug/layer137_out.bin"};
    std::string c0_bin = bin_path + "/layers/c0.bin";
    std::string c2_bin = bin_path + "/layers/c2.bin";
    std::string c4_bin = bin_path + "/layers/c4.bin";
    std::string c5_bin = bin_path + "/layers/c5.bin";
    std::string c6_bin = bin_path + "/layers/c6.bin";
    std::string c7_bin = bin_path + "/layers/c7.bin";
    std::string c9_bin = bin_path + "/layers/c9.bin";
    std::string c10_bin = bin_path + "/layers/c10.bin";
    std::string c11_bin = bin_path + "/layers/c11.bin";
    std::string c13_bin = bin_path + "/layers/c13.bin";
    std::string c14_bin = bin_path + "/layers/c14.bin";
    std::string c15_bin = bin_path + "/layers/c15.bin";
    std::string c17_bin = bin_path + "/layers/c17.bin";
    std::string c19_bin = bin_path + "/layers/c19.bin";
    std::string c20_bin = bin_path + "/layers/c20.bin";
    std::string c21_bin = bin_path + "/layers/c21.bin";
    std::string c23_bin = bin_path + "/layers/c23.bin";
    std::string c24_bin = bin_path + "/layers/c24.bin";
    std::string c25_bin = bin_path + "/layers/c25.bin";
    std::string c26_bin = bin_path + "/layers/c26.bin";
    std::string c28_bin = bin_path + "/layers/c28.bin";
    std::string c29_bin = bin_path + "/layers/c29.bin";
    std::string c30_bin = bin_path + "/layers/c30.bin";
    std::string c32_bin = bin_path + "/layers/c32.bin";
    std::string c33_bin = bin_path + "/layers/c33.bin";
    std::string c34_bin = bin_path + "/layers/c34.bin";
    std::string c36_bin = bin_path + "/layers/c36.bin";
    std::string c38_bin = bin_path + "/layers/c38.bin";
    std::string c39_bin = bin_path + "/layers/c39.bin";
    std::string c40_bin = bin_path + "/layers/c40.bin";
    std::string c42_bin = bin_path + "/layers/c42.bin";
    std::string c43_bin = bin_path + "/layers/c43.bin";
    std::string c44_bin = bin_path + "/layers/c44.bin";
    std::string c45_bin = bin_path + "/layers/c45.bin";
    std::string c47_bin = bin_path + "/layers/c47.bin";
    std::string c48_bin = bin_path + "/layers/c48.bin";
    std::string c49_bin = bin_path + "/layers/c49.bin";
    std::string c51_bin = bin_path + "/layers/c51.bin";
    std::string c52_bin = bin_path + "/layers/c52.bin";
    std::string c53_bin = bin_path + "/layers/c53.bin";
    std::string c55_bin = bin_path + "/layers/c55.bin";
    std::string c56_bin = bin_path + "/layers/c56.bin";
    std::string c57_bin = bin_path + "/layers/c57.bin";
    std::string c59_bin = bin_path + "/layers/c59.bin";
    std::string c60_bin = bin_path + "/layers/c60.bin";
    std::string c61_bin = bin_path + "/layers/c61.bin";
    std::string c63_bin = bin_path + "/layers/c63.bin";
    std::string c65_bin = bin_path + "/layers/c65.bin";
    std::string c66_bin = bin_path + "/layers/c66.bin";
    std::string c67_bin = bin_path + "/layers/c67.bin";
    std::string c69_bin = bin_path + "/layers/c69.bin";
    std::string c70_bin = bin_path + "/layers/c70.bin";
    std::string c71_bin = bin_path + "/layers/c71.bin";
    std::string c72_bin = bin_path + "/layers/c72.bin";
    std::string c74_bin = bin_path + "/layers/c74.bin";
    std::string c75_bin = bin_path + "/layers/c75.bin";
    std::string c76_bin = bin_path + "/layers/c76.bin";
    std::string c78_bin = bin_path + "/layers/c78.bin";
    std::string c80_bin = bin_path + "/layers/c80.bin";
    std::string c81_bin = bin_path + "/layers/c81.bin";
    std::string c82_bin = bin_path + "/layers/c82.bin";
    std::string c83_bin = bin_path + "/layers/c83.bin";
    std::string c90_bin = bin_path + "/layers/c90.bin";
    std::string c91_bin = bin_path + "/layers/c91.bin";
    std::string c92_bin = bin_path + "/layers/c92.bin";
    std::string c93_bin = bin_path + "/layers/c93.bin";
    std::string c96_bin = bin_path + "/layers/c96.bin";
    std::string c98_bin = bin_path + "/layers/c98.bin";
    std::string c99_bin = bin_path + "/layers/c99.bin";
    std::string c100_bin = bin_path + "/layers/c100.bin";
    std::string c101_bin = bin_path + "/layers/c101.bin";
    std::string c102_bin = bin_path + "/layers/c102.bin";
    std::string c103_bin = bin_path + "/layers/c103.bin";
    std::string c106_bin = bin_path + "/layers/c106.bin";
    std::string c108_bin = bin_path + "/layers/c108.bin";
    std::string c109_bin = bin_path + "/layers/c109.bin";
    std::string c110_bin = bin_path + "/layers/c110.bin";
    std::string c111_bin = bin_path + "/layers/c111.bin";
    std::string c112_bin = bin_path + "/layers/c112.bin";
    std::string c113_bin = bin_path + "/layers/c113.bin";
    std::string c114_bin = bin_path + "/layers/c114.bin";
    std::string c117_bin = bin_path + "/layers/c117.bin";
    std::string c119_bin = bin_path + "/layers/c119.bin";
    std::string c120_bin = bin_path + "/layers/c120.bin";
    std::string c121_bin = bin_path + "/layers/c121.bin";
    std::string c122_bin = bin_path + "/layers/c122.bin";
    std::string c123_bin = bin_path + "/layers/c123.bin";
    std::string c124_bin = bin_path + "/layers/c124.bin";
    std::string c125_bin = bin_path + "/layers/c125.bin";
    std::string c128_bin = bin_path + "/layers/c128.bin";
    std::string c130_bin = bin_path + "/layers/c130.bin";
    std::string c131_bin = bin_path + "/layers/c131.bin";
    std::string c132_bin = bin_path + "/layers/c132.bin";
    std::string c133_bin = bin_path + "/layers/c133.bin";
    std::string c134_bin = bin_path + "/layers/c134.bin";
    std::string c135_bin = bin_path + "/layers/c135.bin";
    std::string c136_bin = bin_path + "/layers/c136.bin";
    std::string g115_bin = bin_path + "/layers/g115.bin";
    std::string g126_bin = bin_path + "/layers/g126.bin";
    std::string g137_bin = bin_path + "/layers/g137.bin";

    tk::dnn::Conv2d c0(&net, 64, 7, 7, 2, 2, 3, 3, c0_bin, true);
    tk::dnn::Activation a0(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Pooling p1(&net, 2, 2, 2, 2, 0,0,  tk::dnn::POOLING_MAX);

    tk::dnn::Conv2d c2(&net, 128, 1, 1, 1, 1, 0, 0, c2_bin, true);
    tk::dnn::Activation a2(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Layer *r3_layers[1] = {&p1};
    tk::dnn::Route r3(&net, r3_layers, 1);

    tk::dnn::Conv2d c4(&net, 64, 1, 1, 1, 1, 0, 0, c4_bin, true);
    tk::dnn::Activation a4(&net, tk::dnn::ACTIVATION_LEAKY);

    // //1-1
    tk::dnn::Conv2d c5(&net, 128, 1, 1, 1, 1, 0, 0, c5_bin, true);
    tk::dnn::Activation a5(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c6(&net, 128, 3, 3, 1, 1, 1, 1, c6_bin, true, false, false, 32, false);
    tk::dnn::Activation a6(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c7(&net, 128, 1, 1, 1, 1, 0, 0, c7_bin, true);

    tk::dnn::Shortcut s8(&net, &a4);
    tk::dnn::Activation a8(&net, tk::dnn::ACTIVATION_LEAKY);

    //1-2
    tk::dnn::Conv2d c9(&net, 128, 1, 1, 1, 1, 0, 0, c9_bin, true);
    tk::dnn::Activation a9(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c10(&net, 128, 3, 3, 1, 1, 1, 1, c10_bin, true, false, false, 32);
    tk::dnn::Activation a10(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c11(&net, 128, 1, 1, 1, 1, 0, 0, c11_bin, true);

    tk::dnn::Shortcut s12(&net, &a8);
    tk::dnn::Activation a12(&net, tk::dnn::ACTIVATION_LEAKY);

    //1-3
    tk::dnn::Conv2d c13(&net, 128, 1, 1, 1, 1, 0, 0, c13_bin, true);
    tk::dnn::Activation a13(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c14(&net, 128, 3, 3, 1, 1, 1, 1, c14_bin, true, false, false, 32);
    tk::dnn::Activation a14(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c15(&net, 128, 1, 1, 1, 1, 0, 0, c15_bin, true);

    tk::dnn::Shortcut s16(&net, &a12);
    tk::dnn::Activation a16(&net, tk::dnn::ACTIVATION_LEAKY);

    // //1-T
    tk::dnn::Conv2d c17(&net, 128, 1, 1, 1, 1, 0, 0, c17_bin, true);
    tk::dnn::Activation a17(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Layer *r18_layers[2] = {&a17, &a2};
    tk::dnn::Route r18(&net, r18_layers, 2);

    tk::dnn::Conv2d c19(&net, 256, 1, 1, 1, 1, 0, 0, c19_bin, true);
    tk::dnn::Activation a19(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c20(&net, 256, 3, 3, 2, 2, 1, 1, c20_bin, true, false, false, 32);
    tk::dnn::Activation a20(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c21(&net, 256, 1, 1, 1, 1, 0, 0, c21_bin, true);

    tk::dnn::Layer *r22_layers[2] = {&a20};
    tk::dnn::Route r22(&net, r22_layers, 1);

    tk::dnn::Conv2d c23(&net, 256, 1, 1, 1, 1, 0, 0, c23_bin, true);

    //2-1
    tk::dnn::Conv2d c24(&net, 256, 1, 1, 1, 1, 0, 0, c24_bin, true);
    tk::dnn::Activation a24(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c25(&net, 256, 3, 3, 1, 1, 1, 1, c25_bin, true, false, false, 32);
    tk::dnn::Activation a25(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c26(&net, 256, 1, 1, 1, 1, 0, 0, c26_bin, true);

    tk::dnn::Shortcut s27(&net, &c23);
    tk::dnn::Activation a27(&net, tk::dnn::ACTIVATION_LEAKY);

    //2-2
    tk::dnn::Conv2d c28(&net, 256, 1, 1, 1, 1, 0, 0, c28_bin, true);
    tk::dnn::Activation a28(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c29(&net, 256, 3, 3, 1, 1, 1, 1, c29_bin, true, false, false, 32);
    tk::dnn::Activation a29(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c30(&net, 256, 1, 1, 1, 1, 0, 0, c30_bin, true);

    tk::dnn::Shortcut s31(&net, &a27);
    tk::dnn::Activation a31(&net, tk::dnn::ACTIVATION_LEAKY);

    //2-3
    tk::dnn::Conv2d c32(&net, 256, 1, 1, 1, 1, 0, 0, c32_bin, true);
    tk::dnn::Activation a32(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c33(&net, 256, 3, 3, 1, 1, 1, 1, c33_bin, true, false, false, 32);
    tk::dnn::Activation a33(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c34(&net, 256, 1, 1, 1, 1, 0, 0, c34_bin, true);

    tk::dnn::Shortcut s35(&net, &a31);
    tk::dnn::Activation a35(&net, tk::dnn::ACTIVATION_LEAKY);

    // //2-T
    tk::dnn::Conv2d c36(&net, 256, 1, 1, 1, 1, 0, 0, c36_bin, true);
    tk::dnn::Activation a36(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Layer *r37_layers[2] = {&a36, &c21};
    tk::dnn::Route r37(&net, r37_layers, 2);

    tk::dnn::Conv2d c38(&net, 512, 1, 1, 1, 1, 0, 0, c38_bin, true);
    tk::dnn::Activation a38(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c39(&net, 512, 3, 3, 2, 2, 1, 1, c39_bin, true, false, false, 32);
    tk::dnn::Activation a39(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c40(&net, 512, 1, 1, 1, 1, 0, 0, c40_bin, true);

    tk::dnn::Layer *r41_layers[2] = {&a39};
    tk::dnn::Route r41(&net, r41_layers, 1);

    tk::dnn::Conv2d c42(&net, 512, 1, 1, 1, 1, 0, 0, c42_bin, true);

    //3-1
    tk::dnn::Conv2d c43(&net, 512, 1, 1, 1, 1, 0, 0, c43_bin, true);
    tk::dnn::Activation a43(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c44(&net, 512, 3, 3, 1, 1, 1, 1, c44_bin, true, false, false, 32);
    tk::dnn::Activation a44(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c45(&net, 512, 1, 1, 1, 1, 0, 0, c45_bin, true);

    tk::dnn::Shortcut s46(&net, &c42);
    tk::dnn::Activation a46(&net, tk::dnn::ACTIVATION_LEAKY);

    //3-2
    tk::dnn::Conv2d c47(&net, 512, 1, 1, 1, 1, 0, 0, c47_bin, true);
    tk::dnn::Activation a47(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c48(&net, 512, 3, 3, 1, 1, 1, 1, c48_bin, true, false, false, 32);
    tk::dnn::Activation a48(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c49(&net, 512, 1, 1, 1, 1, 0, 0, c49_bin, true);

    tk::dnn::Shortcut s50(&net, &a46);
    tk::dnn::Activation a50(&net, tk::dnn::ACTIVATION_LEAKY);

    //3-3
    tk::dnn::Conv2d c51(&net, 512, 1, 1, 1, 1, 0, 0, c51_bin, true);
    tk::dnn::Activation a51(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c52(&net, 512, 3, 3, 1, 1, 1, 1, c52_bin, true, false, false, 32);
    tk::dnn::Activation a52(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c53(&net, 512, 1, 1, 1, 1, 0, 0, c53_bin, true);

    tk::dnn::Shortcut s54(&net, &a50);
    tk::dnn::Activation a54(&net, tk::dnn::ACTIVATION_LEAKY);

    //3-4
    tk::dnn::Conv2d c55(&net, 512, 1, 1, 1, 1, 0, 0, c55_bin, true);
    tk::dnn::Activation a55(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c56(&net, 512, 3, 3, 1, 1, 1, 1, c56_bin, true, false, false, 32);
    tk::dnn::Activation a56(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c57(&net, 512, 1, 1, 1, 1, 0, 0, c57_bin, true);

    tk::dnn::Shortcut s58(&net, &a54);
    tk::dnn::Activation a58(&net, tk::dnn::ACTIVATION_LEAKY);

    //3-5
    tk::dnn::Conv2d c59(&net, 512, 1, 1, 1, 1, 0, 0, c59_bin, true);
    tk::dnn::Activation a59(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c60(&net, 512, 3, 3, 1, 1, 1, 1, c60_bin, true, false, false, 32);
    tk::dnn::Activation a60(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c61(&net, 512, 1, 1, 1, 1, 0, 0, c61_bin, true);

    tk::dnn::Shortcut s62(&net, &a58);
    tk::dnn::Activation a62(&net, tk::dnn::ACTIVATION_LEAKY);

    //3-T
    tk::dnn::Conv2d c63(&net, 512, 1, 1, 1, 1, 0, 0, c63_bin, true);
    tk::dnn::Activation a63(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Layer *r64_layers[2] = {&a63, &c40};
    tk::dnn::Route r64(&net, r64_layers, 2);

    tk::dnn::Conv2d c65(&net, 1024, 1, 1, 1, 1, 0, 0, c65_bin, true);
    tk::dnn::Activation a65(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c66(&net, 1024, 3, 3, 2, 2, 1, 1, c66_bin, true, false, false, 32);
    tk::dnn::Activation a66(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c67(&net, 1024, 1, 1, 1, 1, 0, 0, c67_bin, true);
    tk::dnn::Activation a67(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Layer *r68_layers[2] = {&a66};
    tk::dnn::Route r68(&net, r68_layers, 1);

    tk::dnn::Conv2d c69(&net, 1024, 1, 1, 1, 1, 0, 0, c69_bin, true);
    tk::dnn::Activation a69(&net, tk::dnn::ACTIVATION_LEAKY);

    //4-1
    tk::dnn::Conv2d c70(&net, 1024, 1, 1, 1, 1, 0, 0, c70_bin, true);
    tk::dnn::Activation a70(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c71(&net, 1024, 3, 3, 1, 1, 1, 1, c71_bin, true, false, false, 32);
    tk::dnn::Activation a71(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c72(&net, 1024, 1, 1, 1, 1, 0, 0, c72_bin, true);

    tk::dnn::Shortcut s73(&net, &a69);
    tk::dnn::Activation a73(&net, tk::dnn::ACTIVATION_LEAKY);

    //4-2
    tk::dnn::Conv2d c74(&net, 1024, 1, 1, 1, 1, 0, 0, c74_bin, true);
    tk::dnn::Activation a74(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c75(&net, 1024, 3, 3, 1, 1, 1, 1, c75_bin, true, false, false, 32);
    tk::dnn::Activation a75(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c76(&net, 1024, 1, 1, 1, 1, 0, 0, c76_bin, true);

    tk::dnn::Shortcut s77(&net, &a73);
    tk::dnn::Activation a77(&net, tk::dnn::ACTIVATION_LEAKY);

    //4-T
    tk::dnn::Conv2d c78(&net, 1024, 1, 1, 1, 1, 0, 0, c78_bin, true);
    tk::dnn::Activation a78(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Layer *r79_layers[2] = {&a78, &a67};
    tk::dnn::Route r79(&net, r79_layers, 2);

    tk::dnn::Conv2d c80(&net, 2048, 1, 1, 1, 1, 0, 0, c80_bin, true);
    tk::dnn::Activation a80(&net, tk::dnn::ACTIVATION_LEAKY);

    // ////////////////////

    tk::dnn::Conv2d c81(&net, 512, 1, 1, 1, 1, 0, 0, c81_bin, true);
    tk::dnn::Activation a81(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c82(&net, 1024, 3, 3, 1, 1, 1, 1, c82_bin, true);
    tk::dnn::Activation a82(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c83(&net, 512, 1, 1, 1, 1, 0, 0, c83_bin, true);
    tk::dnn::Activation a83(&net, tk::dnn::ACTIVATION_LEAKY);

    //SPP
    tk::dnn::Pooling p84(&net, 5, 5, 1, 1,0,0, tk::dnn::POOLING_MAX, false, true);
    tk::dnn::Layer *r85_layers[1] = {&a83};
    tk::dnn::Route r85(&net, r85_layers, 1);

    tk::dnn::Pooling p86(&net, 9, 9, 1, 1,0,0, tk::dnn::POOLING_MAX, false, true);
    tk::dnn::Layer *r87_layers[1] = {&a83};
    tk::dnn::Route r87(&net, r87_layers, 1);

    tk::dnn::Pooling p88(&net, 13, 13, 1, 1, 12, 12, tk::dnn::POOLING_MAX, false, true);
    tk::dnn::Layer *r89_layers[4] = {&p88, &p86, &p84, &a83};
    tk::dnn::Route r89(&net, r89_layers, 4);
    //END SPP

    tk::dnn::Conv2d c90(&net, 512, 1, 1, 1, 1, 0, 0, c90_bin, true);
    tk::dnn::Activation a90(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c91(&net, 1024, 3, 3, 1, 1, 1, 1, c91_bin, true);
    tk::dnn::Activation a91(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c92(&net, 512, 1, 1, 1, 1, 0, 0, c92_bin, true);
    tk::dnn::Activation a92(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c93(&net, 256, 1, 1, 1, 1, 0, 0, c93_bin, true);
    tk::dnn::Activation a93(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Upsample u94(&net, 2);
    tk::dnn::Layer *r95_layers[1] = {&a65};
    tk::dnn::Route r95(&net, r95_layers, 1);
    tk::dnn::Conv2d c96(&net, 256, 1, 1, 1, 1, 0, 0, c96_bin, true);
    tk::dnn::Activation a96(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer *r97_layers[2] = {&a96,&u94};
    tk::dnn::Route r97(&net, r97_layers, 2);

    tk::dnn::Conv2d c98(&net, 256, 1, 1, 1, 1, 0, 0, c98_bin, true);
    tk::dnn::Activation a98(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c99(&net, 512, 3, 3, 1, 1, 1, 1, c99_bin, true);
    tk::dnn::Activation a99(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c100(&net, 256, 1, 1, 1, 1, 0, 0, c100_bin, true);
    tk::dnn::Activation a100(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c101(&net, 512, 3, 3, 1, 1, 1, 1, c101_bin, true);
    tk::dnn::Activation a101(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c102(&net, 256, 1, 1, 1, 1, 0, 0, c102_bin, true);
    tk::dnn::Activation a102(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c103(&net, 128, 1, 1, 1, 1, 0, 0, c103_bin, true);
    tk::dnn::Activation a103(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Upsample u104(&net, 2);
    tk::dnn::Layer *r105_layers[1] = {&a38};
    tk::dnn::Route r105(&net, r105_layers, 1);
    tk::dnn::Conv2d c106(&net, 128, 1, 1, 1, 1, 0, 0, c106_bin, true);
    tk::dnn::Activation a106(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer *r107_layers[2] = {&a106,&u104};
    tk::dnn::Route r107(&net, r107_layers, 2);


    tk::dnn::Conv2d c108(&net, 128, 1, 1, 1, 1, 0, 0, c108_bin, true);
    tk::dnn::Activation a108(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c109(&net, 256, 3, 3, 1, 1, 1, 1, c109_bin, true);
    tk::dnn::Activation a109(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c110(&net, 128, 1, 1, 1, 1, 0, 0, c110_bin, true);
    tk::dnn::Activation a110(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c111(&net, 256, 3, 3, 1, 1, 1, 1, c111_bin, true);
    tk::dnn::Activation a111(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c112(&net, 128, 1, 1, 1, 1, 0, 0, c112_bin, true);
    tk::dnn::Activation a112(&net, tk::dnn::ACTIVATION_LEAKY);

    // ###########################

    tk::dnn::Conv2d c113(&net, 256, 3, 3, 1, 1, 1, 1, c113_bin, true);
    tk::dnn::Activation a113(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c114(&net, 255, 1, 1, 1, 1, 0, 0, c114_bin, false);
    tk::dnn::Yolo yolo115(&net, classes, 3, g115_bin);

    tk::dnn::Layer *r116_layers[1] = {&a112};
    tk::dnn::Route r116(&net, r116_layers, 1);
    tk::dnn::Conv2d c117(&net, 256, 3, 3, 2, 2, 1, 1, c117_bin, true);
    tk::dnn::Activation a117(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer *r118_layers[2] = {&a117,&a102};
    tk::dnn::Route r118(&net, r118_layers, 2);

    tk::dnn::Conv2d c119(&net, 256, 1, 1, 1, 1, 0, 0, c119_bin, true);
    tk::dnn::Activation a119(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c120(&net, 512, 3, 3, 1, 1, 1, 1, c120_bin, true);
    tk::dnn::Activation a120(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c121(&net, 256, 1, 1, 1, 1, 0, 0, c121_bin, true);
    tk::dnn::Activation a121(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c122(&net, 512, 3, 3, 1, 1, 1, 1, c122_bin, true);
    tk::dnn::Activation a122(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c123(&net, 256, 1, 1, 1, 1, 0, 0, c123_bin, true);
    tk::dnn::Activation a123(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Conv2d c124(&net, 512, 3, 3, 1, 1, 1, 1, c124_bin, true);
    tk::dnn::Activation a124(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c125(&net, 255, 1, 1, 1, 1, 0, 0, c125_bin, false);
    tk::dnn::Yolo yolo126(&net, classes, 3, g126_bin);

    tk::dnn::Layer *r127_layers[1] = {&a123};
    tk::dnn::Route r127(&net, r127_layers, 1);
    tk::dnn::Conv2d c128(&net, 512, 3, 3, 2, 2, 1, 1, c128_bin, true);
    tk::dnn::Activation a128(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer *r129_layers[2] = {&a128,&a92};
    tk::dnn::Route r129(&net, r129_layers, 2);

    tk::dnn::Conv2d c130(&net, 512, 1, 1, 1, 1, 0, 0, c130_bin, true);
    tk::dnn::Activation a130(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c131(&net, 1024, 3, 3, 1, 1, 1, 1, c131_bin, true);
    tk::dnn::Activation a131(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c132(&net, 512, 1, 1, 1, 1, 0, 0, c132_bin, true);
    tk::dnn::Activation a132(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c133(&net, 1024, 3, 3, 1, 1, 1, 1, c133_bin, true);
    tk::dnn::Activation a133(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c134(&net, 512, 1, 1, 1, 1, 0, 0, c134_bin, true);
    tk::dnn::Activation a134(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Conv2d c135(&net, 1024, 3, 3, 1, 1, 1, 1, c135_bin, true);
    tk::dnn::Activation a135(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c136(&net, 255, 1, 1, 1, 1, 0, 0, c136_bin, false);
    tk::dnn::Yolo yolo137(&net, classes, 3, g137_bin);

    yolo[0] = &yolo115;
    yolo[1] = &yolo126;
    yolo[2] = &yolo137;

    // fill classes names
    for (int i = 0; i < 3; i++)
    {
        yolo[i]->classesNames = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    }

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    //print network model
    net.print();

    // //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, "csresnext50-panet-spp.rt");

    // the network have 3 outputs
    tk::dnn::dataDim_t out_dim[3];
    for (int i = 0; i < 3; i++)
        out_dim[i] = yolo[i]->output_dim;
    dnnType *cudnn_out[3], *rt_out[3];

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30);
    {
        dim1.print();
        TIMER_START
        net.infer(dim1, data);
        TIMER_STOP
        dim1.print();
    }
    
    for (int i = 0; i < 3; i++)
        cudnn_out[i] = yolo[i]->dstData;

    printCenteredTitle(" compute detections ", '=', 30);
    TIMER_START
    int ndets = 0;
    tk::dnn::Yolo::detection *dets = tk::dnn::Yolo::allocateDetections(tk::dnn::Yolo::MAX_DETECTIONS, classes);
    for (int i = 0; i < 3; i++)
        yolo[i]->computeDetections(dets, ndets, net.input_dim.w, net.input_dim.h, 0.5);
    tk::dnn::Yolo::mergeDetections(dets, ndets, classes);

    for (int j = 0; j < ndets; j++)
    {
        tk::dnn::Yolo::box b = dets[j].bbox;
        int x0 = (b.x - b.w / 2.);
        int x1 = (b.x + b.w / 2.);
        int y0 = (b.y - b.h / 2.);
        int y1 = (b.y + b.h / 2.);

        int cl = 0;
        for (int c = 0; c < classes; ++c)
        {
            float prob = dets[j].prob[c];
            if (prob > 0)
                cl = c;
        }
        std::cout << cl << ": " << x0 << " " << y0 << " " << x1 << " " << y1 << "\n";
    }
    TIMER_STOP

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30);
    {
        dim2.print();
        TIMER_START
        netRT.infer(dim2, data);
        TIMER_STOP
        dim2.print();
    }

    for (int i = 0; i < 3; i++)
        rt_out[i] = (dnnType *)netRT.buffersRT[i + 1];

    for (int i = 0; i < 3; i++)
    {
        printCenteredTitle((std::string(" YOLO ") + std::to_string(i) + " CHECK RESULTS ").c_str(), '=', 30);
        dnnType *out, *out_h;
        int odim = out_dim[i].tot();
        readBinaryFile(output_bins[i], odim, &out_h, &out);
        std::cout << "CUDNN vs correct";
        checkResult(odim, cudnn_out[i], out);
        std::cout << "TRT   vs correct";
        checkResult(odim, rt_out[i], out);
        std::cout << "CUDNN vs TRT    ";
        checkResult(odim, cudnn_out[i], rt_out[i]);
    }
    return 0;
}
