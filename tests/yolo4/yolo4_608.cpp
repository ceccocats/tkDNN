#include <iostream>
#include <vector>
#include "tkdnn.h"

int main()
{

    // Network layout
    tk::dnn::dataDim_t dim(1, 3, 608, 608, 1);
    tk::dnn::Network net(dim);

    // create yolo4_608 model
    std::string bin_path = "yolo4_608";
    int classes = 80;
    tk::dnn::Yolo *yolo[3];

    std::string input_bin = bin_path + "/layers/input.bin";
    
    std::vector<std::string> output_bins = {
        bin_path + "/debug/layer139_out.bin",
        bin_path + "/debug/layer150_out.bin",
        bin_path + "/debug/layer161_out.bin"};
    std::string c0_bin = bin_path + "/layers/c0.bin";
    std::string c1_bin = bin_path + "/layers/c1.bin";
    std::string c2_bin = bin_path + "/layers/c2.bin";
    std::string c3_bin = bin_path + "/layers/c3.bin";
    std::string c4_bin = bin_path + "/layers/c4.bin";
    std::string c5_bin = bin_path + "/layers/c5.bin";
    std::string c6_bin = bin_path + "/layers/c6.bin";
    std::string c7_bin = bin_path + "/layers/c7.bin";
    std::string c8_bin = bin_path + "/layers/c8.bin";
    std::string c10_bin = bin_path + "/layers/c10.bin";
    std::string c11_bin = bin_path + "/layers/c11.bin";
    std::string c12_bin = bin_path + "/layers/c12.bin";
    std::string c13_bin = bin_path + "/layers/c13.bin";
    std::string c14_bin = bin_path + "/layers/c14.bin";
    std::string c15_bin = bin_path + "/layers/c15.bin";
    std::string c16_bin = bin_path + "/layers/c16.bin";
    std::string c17_bin = bin_path + "/layers/c17.bin";
    std::string c18_bin = bin_path + "/layers/c18.bin";
    std::string c19_bin = bin_path + "/layers/c19.bin";
    std::string c20_bin = bin_path + "/layers/c20.bin";
    std::string c21_bin = bin_path + "/layers/c21.bin";
    std::string c23_bin = bin_path + "/layers/c23.bin";
    std::string c24_bin = bin_path + "/layers/c24.bin";
    std::string c25_bin = bin_path + "/layers/c25.bin";
    std::string c26_bin = bin_path + "/layers/c26.bin";
    std::string c27_bin = bin_path + "/layers/c27.bin";
    std::string c28_bin = bin_path + "/layers/c28.bin";
    std::string c29_bin = bin_path + "/layers/c29.bin";
    std::string c30_bin = bin_path + "/layers/c30.bin";
    std::string c31_bin = bin_path + "/layers/c31.bin";
    std::string c32_bin = bin_path + "/layers/c32.bin";
    std::string c33_bin = bin_path + "/layers/c33.bin";
    std::string c34_bin = bin_path + "/layers/c34.bin";
    std::string c35_bin = bin_path + "/layers/c35.bin";
    std::string c36_bin = bin_path + "/layers/c36.bin";
    std::string c37_bin = bin_path + "/layers/c37.bin";
    std::string c38_bin = bin_path + "/layers/c38.bin";
    std::string c39_bin = bin_path + "/layers/c39.bin";
    std::string c40_bin = bin_path + "/layers/c40.bin";
    std::string c41_bin = bin_path + "/layers/c41.bin";
    std::string c42_bin = bin_path + "/layers/c42.bin";
    std::string c43_bin = bin_path + "/layers/c43.bin";
    std::string c44_bin = bin_path + "/layers/c44.bin";
    std::string c45_bin = bin_path + "/layers/c45.bin";
    std::string c46_bin = bin_path + "/layers/c46.bin";
    std::string c47_bin = bin_path + "/layers/c47.bin";
    std::string c48_bin = bin_path + "/layers/c48.bin";
    std::string c49_bin = bin_path + "/layers/c49.bin";
    std::string c50_bin = bin_path + "/layers/c50.bin";
    std::string c51_bin = bin_path + "/layers/c51.bin";
    std::string c52_bin = bin_path + "/layers/c52.bin";
    std::string c53_bin = bin_path + "/layers/c53.bin";
    std::string c54_bin = bin_path + "/layers/c54.bin";
    std::string c55_bin = bin_path + "/layers/c55.bin";
    std::string c56_bin = bin_path + "/layers/c56.bin";
    std::string c57_bin = bin_path + "/layers/c57.bin";
    std::string c58_bin = bin_path + "/layers/c58.bin";
    std::string c59_bin = bin_path + "/layers/c59.bin";
    std::string c60_bin = bin_path + "/layers/c60.bin";
    std::string c61_bin = bin_path + "/layers/c61.bin";
    std::string c62_bin = bin_path + "/layers/c62.bin";
    std::string c63_bin = bin_path + "/layers/c63.bin";
    std::string c65_bin = bin_path + "/layers/c65.bin";
    std::string c66_bin = bin_path + "/layers/c66.bin";
    std::string c67_bin = bin_path + "/layers/c67.bin";
    std::string c68_bin = bin_path + "/layers/c68.bin";
    std::string c69_bin = bin_path + "/layers/c69.bin";
    std::string c70_bin = bin_path + "/layers/c70.bin";
    std::string c71_bin = bin_path + "/layers/c71.bin";
    std::string c72_bin = bin_path + "/layers/c72.bin";
    std::string c74_bin = bin_path + "/layers/c74.bin";
    std::string c75_bin = bin_path + "/layers/c75.bin";
    std::string c76_bin = bin_path + "/layers/c76.bin";
    std::string c77_bin = bin_path + "/layers/c77.bin";
    std::string c78_bin = bin_path + "/layers/c78.bin";
    std::string c80_bin = bin_path + "/layers/c80.bin";
    std::string c81_bin = bin_path + "/layers/c81.bin";
    std::string c82_bin = bin_path + "/layers/c82.bin";
    std::string c83_bin = bin_path + "/layers/c83.bin";
    std::string c85_bin = bin_path + "/layers/c85.bin";
    std::string c86_bin = bin_path + "/layers/c86.bin";
    std::string c87_bin = bin_path + "/layers/c87.bin";
    std::string c89_bin = bin_path + "/layers/c89.bin";
    std::string c90_bin = bin_path + "/layers/c90.bin";
    std::string c91_bin = bin_path + "/layers/c91.bin";
    std::string c92_bin = bin_path + "/layers/c92.bin";
    std::string c93_bin = bin_path + "/layers/c93.bin";
    std::string c94_bin = bin_path + "/layers/c94.bin";
    std::string c96_bin = bin_path + "/layers/c96.bin";
    std::string c97_bin = bin_path + "/layers/c97.bin";
    std::string c98_bin = bin_path + "/layers/c98.bin";
    std::string c99_bin = bin_path + "/layers/c99.bin";
    std::string c100_bin = bin_path + "/layers/c100.bin";
    std::string c101_bin = bin_path + "/layers/c101.bin";
    std::string c102_bin = bin_path + "/layers/c102.bin";
    std::string c103_bin = bin_path + "/layers/c103.bin";
    std::string c104_bin = bin_path + "/layers/c104.bin";
    std::string c105_bin = bin_path + "/layers/c105.bin";
    std::string c106_bin = bin_path + "/layers/c106.bin";
    std::string c107_bin = bin_path + "/layers/c107.bin";
    std::string c108_bin = bin_path + "/layers/c108.bin";
    std::string c109_bin = bin_path + "/layers/c109.bin";
    std::string c110_bin = bin_path + "/layers/c110.bin";
    std::string c111_bin = bin_path + "/layers/c111.bin";
    std::string c112_bin = bin_path + "/layers/c112.bin";
    std::string c113_bin = bin_path + "/layers/c113.bin";
    std::string c114_bin = bin_path + "/layers/c114.bin";
    std::string c115_bin = bin_path + "/layers/c115.bin";
    std::string c116_bin = bin_path + "/layers/c116.bin";
    std::string c117_bin = bin_path + "/layers/c117.bin";
    std::string c119_bin = bin_path + "/layers/c119.bin";
    std::string c120_bin = bin_path + "/layers/c120.bin";
    std::string c121_bin = bin_path + "/layers/c121.bin";
    std::string c122_bin = bin_path + "/layers/c122.bin";
    std::string c123_bin = bin_path + "/layers/c123.bin";
    std::string c124_bin = bin_path + "/layers/c124.bin";
    std::string c125_bin = bin_path + "/layers/c125.bin";
    std::string c126_bin = bin_path + "/layers/c126.bin";
    std::string c127_bin = bin_path + "/layers/c127.bin";
    std::string c128_bin = bin_path + "/layers/c128.bin";
    std::string c130_bin = bin_path + "/layers/c130.bin";
    std::string c131_bin = bin_path + "/layers/c131.bin";
    std::string c132_bin = bin_path + "/layers/c132.bin";
    std::string c133_bin = bin_path + "/layers/c133.bin";
    std::string c134_bin = bin_path + "/layers/c134.bin";
    std::string c135_bin = bin_path + "/layers/c135.bin";
    std::string c136_bin = bin_path + "/layers/c136.bin";
    std::string c137_bin = bin_path + "/layers/c137.bin";
    std::string c138_bin = bin_path + "/layers/c138.bin";
    std::string c141_bin = bin_path + "/layers/c141.bin";
    std::string c142_bin = bin_path + "/layers/c142.bin";
    std::string c143_bin = bin_path + "/layers/c143.bin";
    std::string c144_bin = bin_path + "/layers/c144.bin";
    std::string c145_bin = bin_path + "/layers/c145.bin";
    std::string c146_bin = bin_path + "/layers/c146.bin";
    std::string c147_bin = bin_path + "/layers/c147.bin";
    std::string c148_bin = bin_path + "/layers/c148.bin";
    std::string c149_bin = bin_path + "/layers/c149.bin";
    std::string c150_bin = bin_path + "/layers/c150.bin";
    std::string c151_bin = bin_path + "/layers/c151.bin";
    std::string c152_bin = bin_path + "/layers/c152.bin";
    std::string c153_bin = bin_path + "/layers/c153.bin";
    std::string c154_bin = bin_path + "/layers/c154.bin";
    std::string c155_bin = bin_path + "/layers/c155.bin";
    std::string c156_bin = bin_path + "/layers/c156.bin";
    std::string c157_bin = bin_path + "/layers/c157.bin";
    std::string c158_bin = bin_path + "/layers/c158.bin";
    std::string c159_bin = bin_path + "/layers/c159.bin";
    std::string c160_bin = bin_path + "/layers/c160.bin";    
    std::string g139_bin = bin_path + "/layers/g139.bin";
    std::string g150_bin = bin_path + "/layers/g150.bin";
    std::string g161_bin = bin_path + "/layers/g161.bin";
    

    downloadWeightsifDoNotExist(input_bin, bin_path, "https://cloud.hipert.unimore.it/s/d97CFzYqCPCp5Hg/download");

    tk::dnn::Conv2d c0(&net, 32, 3, 3, 1, 1, 1, 1, c0_bin, true);
    tk::dnn::Activation a0(&net, tk::dnn::ACTIVATION_MISH);

    // downsample
    tk::dnn::Conv2d c1(&net, 64, 3, 3, 2, 2, 1, 1, c1_bin, true);
    tk::dnn::Activation a1(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c2(&net, 64, 1, 1, 1, 1, 0, 0, c2_bin, true);
    tk::dnn::Activation a2(&net, tk::dnn::ACTIVATION_MISH);  

    tk::dnn::Layer *r3_layers[1] = {&a1};
    tk::dnn::Route r3(&net, r3_layers, 1);
    
    tk::dnn::Conv2d c4(&net, 64, 1, 1, 1, 1, 0, 0, c4_bin, true);
    tk::dnn::Activation a4(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c5(&net, 32, 1, 1, 1, 1, 0, 0, c5_bin, true);
    tk::dnn::Activation a5(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c6(&net, 64, 3, 3, 1, 1, 1, 1, c6_bin, true);
    tk::dnn::Activation a6(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s7(&net, &a4);

    tk::dnn::Conv2d c8(&net, 64, 1, 1, 1, 1, 0, 0, c8_bin, true);
    tk::dnn::Activation a8(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Layer *r9_layers[2] = {&a8, &a2};
    tk::dnn::Route r9(&net, r9_layers, 2);

    tk::dnn::Conv2d c10(&net, 64, 1, 1, 1, 1, 0, 0, c10_bin, true);
    tk::dnn::Activation a10(&net, tk::dnn::ACTIVATION_MISH);

    // downsample
    tk::dnn::Conv2d c11(&net, 128, 3, 3, 2, 2, 1, 1, c11_bin, true);
    tk::dnn::Activation a11(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c12(&net, 64, 1, 1, 1, 1, 0, 0, c12_bin, true);
    tk::dnn::Activation a12(&net, tk::dnn::ACTIVATION_MISH);  

    tk::dnn::Layer *r13_layers[1] = {&a11};
    tk::dnn::Route r13(&net, r13_layers, 1);
    
    tk::dnn::Conv2d c14(&net, 64, 1, 1, 1, 1, 0, 0, c14_bin, true);
    tk::dnn::Activation a14(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c15(&net, 64, 1, 1, 1, 1, 0, 0, c15_bin, true);
    tk::dnn::Activation a15(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c16(&net, 64, 3, 3, 1, 1, 1, 1, c16_bin, true);
    tk::dnn::Activation a16(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s17(&net, &a14);

    tk::dnn::Conv2d c18(&net, 64, 1, 1, 1, 1, 0, 0, c18_bin, true);
    tk::dnn::Activation a18(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c19(&net, 64, 3, 3, 1, 1, 1, 1, c19_bin, true);
    tk::dnn::Activation a19(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s20(&net, &s17);

    tk::dnn::Conv2d c21(&net, 64, 1, 1, 1, 1, 0, 0, c21_bin, true);
    tk::dnn::Activation a21(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Layer *r22_layers[2] = {&a21, &a12};
    tk::dnn::Route r22(&net, r22_layers, 2);

    tk::dnn::Conv2d c23(&net, 128, 1, 1, 1, 1, 0, 0, c23_bin, true);
    tk::dnn::Activation a23(&net, tk::dnn::ACTIVATION_MISH);

    //downsample
    tk::dnn::Conv2d c24(&net, 256, 3, 3, 2, 2, 1, 1, c24_bin, true);
    tk::dnn::Activation a24(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c25(&net, 128, 1, 1, 1, 1, 0, 0, c25_bin, true);
    tk::dnn::Activation a25(&net, tk::dnn::ACTIVATION_MISH);  

    tk::dnn::Layer *r26_layers[1] = {&a24};
    tk::dnn::Route r26(&net, r26_layers, 1);
    
    tk::dnn::Conv2d c27(&net, 128, 1, 1, 1, 1, 0, 0, c27_bin, true);
    tk::dnn::Activation a27(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c28(&net, 128, 1, 1, 1, 1, 0, 0, c28_bin, true);
    tk::dnn::Activation a28(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c29(&net, 128, 3, 3, 1, 1, 1, 1, c29_bin, true);
    tk::dnn::Activation a29(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s30(&net, &a27);

    tk::dnn::Conv2d c31(&net, 128, 1, 1, 1, 1, 0, 0, c31_bin, true);
    tk::dnn::Activation a31(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c32(&net, 128, 3, 3, 1, 1, 1, 1, c32_bin, true);
    tk::dnn::Activation a32(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s33(&net, &s30);

    tk::dnn::Conv2d c34(&net, 128, 1, 1, 1, 1, 0, 0, c34_bin, true);
    tk::dnn::Activation a34(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c35(&net, 128, 3, 3, 1, 1, 1, 1, c35_bin, true);
    tk::dnn::Activation a35(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s36(&net, &s33);

    tk::dnn::Conv2d c37(&net, 128, 1, 1, 1, 1, 0, 0, c37_bin, true);
    tk::dnn::Activation a37(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c38(&net, 128, 3, 3, 1, 1, 1, 1, c38_bin, true);
    tk::dnn::Activation a38(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s39(&net, &s36);
    
    tk::dnn::Conv2d c40(&net, 128, 1, 1, 1, 1, 0, 0, c40_bin, true);
    tk::dnn::Activation a40(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c41(&net, 128, 3, 3, 1, 1, 1, 1, c41_bin, true);
    tk::dnn::Activation a41(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s42(&net, &s39);

    tk::dnn::Conv2d c43(&net, 128, 1, 1, 1, 1, 0, 0, c43_bin, true);
    tk::dnn::Activation a43(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c44(&net, 128, 3, 3, 1, 1, 1, 1, c44_bin, true);
    tk::dnn::Activation a44(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s45(&net, &s42);

    tk::dnn::Conv2d c46(&net, 128, 1, 1, 1, 1, 0, 0, c46_bin, true);
    tk::dnn::Activation a46(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c47(&net, 128, 3, 3, 1, 1, 1, 1, c47_bin, true);
    tk::dnn::Activation a47(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s48(&net, &s45);

    tk::dnn::Conv2d c49(&net, 128, 1, 1, 1, 1, 0, 0, c49_bin, true);
    tk::dnn::Activation a49(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c50(&net, 128, 3, 3, 1, 1, 1, 1, c50_bin, true);
    tk::dnn::Activation a50(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s51(&net, &s48);

    tk::dnn::Conv2d c52(&net, 128, 1, 1, 1, 1, 0, 0, c52_bin, true);
    tk::dnn::Activation a52(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Layer *r53_layers[2] = {&a52, &a25};
    tk::dnn::Route r53(&net, r53_layers, 2);

    tk::dnn::Conv2d c54(&net, 256, 1, 1, 1, 1, 0, 0, c54_bin, true);
    tk::dnn::Activation a54(&net, tk::dnn::ACTIVATION_MISH);

    //downsample
    tk::dnn::Conv2d c55(&net, 512, 3, 3, 2, 2, 1, 1, c55_bin, true);
    tk::dnn::Activation a55(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c56(&net, 256, 1, 1, 1, 1, 0, 0, c56_bin, true);
    tk::dnn::Activation a56(&net, tk::dnn::ACTIVATION_MISH);  

    tk::dnn::Layer *r57_layers[1] = {&a55};
    tk::dnn::Route r57(&net, r57_layers, 1);
    
    tk::dnn::Conv2d c58(&net, 256, 1, 1, 1, 1, 0, 0, c58_bin, true);
    tk::dnn::Activation a58(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c59(&net, 256, 1, 1, 1, 1, 0, 0, c59_bin, true);
    tk::dnn::Activation a59(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c60(&net, 256, 3, 3, 1, 1, 1, 1, c60_bin, true);
    tk::dnn::Activation a60(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s61(&net, &a58);

    tk::dnn::Conv2d c62(&net, 256, 1, 1, 1, 1, 0, 0, c62_bin, true);
    tk::dnn::Activation a62(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c63(&net, 256, 3, 3, 1, 1, 1, 1, c63_bin, true);
    tk::dnn::Activation a63(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s64(&net, &s61);

    tk::dnn::Conv2d c65(&net, 256, 1, 1, 1, 1, 0, 0, c65_bin, true);
    tk::dnn::Activation a65(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c66(&net, 256, 3, 3, 1, 1, 1, 1, c66_bin, true);
    tk::dnn::Activation a66(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s67(&net, &s64);

    tk::dnn::Conv2d c68(&net, 256, 1, 1, 1, 1, 0, 0, c68_bin, true);
    tk::dnn::Activation a68(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c69(&net, 256, 3, 3, 1, 1, 1, 1, c69_bin, true);
    tk::dnn::Activation a69(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s70(&net, &s67);
    
    tk::dnn::Conv2d c71(&net, 256, 1, 1, 1, 1, 0, 0, c71_bin, true);
    tk::dnn::Activation a71(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c72(&net, 256, 3, 3, 1, 1, 1, 1, c72_bin, true);
    tk::dnn::Activation a72(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s73(&net, &s70);

    tk::dnn::Conv2d c74(&net, 256, 1, 1, 1, 1, 0, 0, c74_bin, true);
    tk::dnn::Activation a74(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c75(&net, 256, 3, 3, 1, 1, 1, 1, c75_bin, true);
    tk::dnn::Activation a75(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s76(&net, &s73);

    tk::dnn::Conv2d c77(&net, 256, 1, 1, 1, 1, 0, 0, c77_bin, true);
    tk::dnn::Activation a77(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c78(&net, 256, 3, 3, 1, 1, 1, 1, c78_bin, true);
    tk::dnn::Activation a78(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s79(&net, &s76);

    tk::dnn::Conv2d c80(&net, 256, 1, 1, 1, 1, 0, 0, c80_bin, true);
    tk::dnn::Activation a80(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c81(&net, 256, 3, 3, 1, 1, 1, 1, c81_bin, true);
    tk::dnn::Activation a81(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s82(&net, &s79);

    tk::dnn::Conv2d c83(&net, 256, 1, 1, 1, 1, 0, 0, c83_bin, true);
    tk::dnn::Activation a83(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Layer *r84_layers[2] = {&a83, &a56};
    tk::dnn::Route r84(&net, r84_layers, 2);

    tk::dnn::Conv2d c85(&net, 512, 1, 1, 1, 1, 0, 0, c85_bin, true);
    tk::dnn::Activation a85(&net, tk::dnn::ACTIVATION_MISH);

    //downsample
    tk::dnn::Conv2d c86(&net, 1024, 3, 3, 2, 2, 1, 1, c86_bin, true);
    tk::dnn::Activation a86(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c87(&net, 512, 1, 1, 1, 1, 0, 0, c87_bin, true);
    tk::dnn::Activation a87(&net, tk::dnn::ACTIVATION_MISH);  

    tk::dnn::Layer *r88_layers[1] = {&a86};
    tk::dnn::Route r88(&net, r88_layers, 1);
    
    tk::dnn::Conv2d c89(&net, 512, 1, 1, 1, 1, 0, 0, c89_bin, true);
    tk::dnn::Activation a89(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c90(&net, 512, 1, 1, 1, 1, 0, 0, c90_bin, true);
    tk::dnn::Activation a90(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c91(&net, 512, 3, 3, 1, 1, 1, 1, c91_bin, true);
    tk::dnn::Activation a91(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s92(&net, &a89);

    tk::dnn::Conv2d c93(&net, 512, 1, 1, 1, 1, 0, 0, c93_bin, true);
    tk::dnn::Activation a93(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c94(&net, 512, 3, 3, 1, 1, 1, 1, c94_bin, true);
    tk::dnn::Activation a94(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s95(&net, &s92);

    tk::dnn::Conv2d c96(&net, 512, 1, 1, 1, 1, 0, 0, c96_bin, true);
    tk::dnn::Activation a96(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c97(&net, 512, 3, 3, 1, 1, 1, 1, c97_bin, true);
    tk::dnn::Activation a97(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s98(&net, &s95);

    tk::dnn::Conv2d c99(&net, 512, 1, 1, 1, 1, 0, 0, c99_bin, true);
    tk::dnn::Activation a99(&net, tk::dnn::ACTIVATION_MISH);
    tk::dnn::Conv2d c100(&net, 512, 3, 3, 1, 1, 1, 1, c100_bin, true);
    tk::dnn::Activation a100(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Shortcut s101(&net, &s98);
    
    tk::dnn::Conv2d c102(&net, 512, 1, 1, 1, 1, 0, 0, c102_bin, true);
    tk::dnn::Activation a102(&net, tk::dnn::ACTIVATION_MISH);

    tk::dnn::Layer *r103_layers[2] = {&a102, &a87};
    tk::dnn::Route r103(&net, r103_layers, 2);

    tk::dnn::Conv2d c104(&net, 1024, 1, 1, 1, 1, 0, 0, c104_bin, true);
    tk::dnn::Activation a104(&net, tk::dnn::ACTIVATION_MISH);


    //################
    tk::dnn::Conv2d c105(&net, 512, 1, 1, 1, 1, 0, 0, c105_bin, true);
    tk::dnn::Activation a105(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c106(&net, 1024, 3, 3, 1, 1, 1, 1, c106_bin, true);
    tk::dnn::Activation a106(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c107(&net, 512, 1, 1, 1, 1, 0, 0, c107_bin, true);
    tk::dnn::Activation a107(&net, tk::dnn::ACTIVATION_LEAKY);

    //SPP
    tk::dnn::Pooling p108(&net, 5, 5, 1, 1, 0, 0, tk::dnn::POOLING_MAX_FIXEDSIZE);
    tk::dnn::Layer *r109_layers[1] = {&a107};
    tk::dnn::Route r109(&net, r109_layers, 1);

    tk::dnn::Pooling p110(&net, 9, 9, 1, 1, 0, 0, tk::dnn::POOLING_MAX_FIXEDSIZE);
    tk::dnn::Layer *r111_layers[1] = {&a107};
    tk::dnn::Route r111(&net, r111_layers, 1);

    tk::dnn::Pooling p112(&net, 13, 13, 1, 1, 12, 12, tk::dnn::POOLING_MAX_FIXEDSIZE);
    tk::dnn::Layer *r113_layers[4] = {&p112, &p110, &p108, &a107};
    tk::dnn::Route r113(&net, r113_layers, 4);
    //END SPP

    tk::dnn::Conv2d c114(&net, 512, 1, 1, 1, 1, 0, 0, c114_bin, true);
    tk::dnn::Activation a114(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c115(&net, 1024, 3, 3, 1, 1, 1, 1, c115_bin, true);
    tk::dnn::Activation a115(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c116(&net, 512, 1, 1, 1, 1, 0, 0, c116_bin, true);
    tk::dnn::Activation a116(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c117(&net, 256, 1, 1, 1, 1, 0, 0, c117_bin, true);
    tk::dnn::Activation a117(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Upsample u118(&net, 2);
    tk::dnn::Layer *r119_layers[1] = {&a85};
    tk::dnn::Route r119(&net, r119_layers, 1);
    tk::dnn::Conv2d c120(&net, 256, 1, 1, 1, 1, 0, 0, c120_bin, true);
    tk::dnn::Activation a120(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer *r121_layers[2] = {&a120,&u118};
    tk::dnn::Route r121(&net, r121_layers, 2);

    tk::dnn::Conv2d c122(&net, 256, 1, 1, 1, 1, 0, 0, c122_bin, true);
    tk::dnn::Activation a122(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c123(&net, 512, 3, 3, 1, 1, 1, 1, c123_bin, true);
    tk::dnn::Activation a123(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c124(&net, 256, 1, 1, 1, 1, 0, 0, c124_bin, true);
    tk::dnn::Activation a124(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c125(&net, 512, 3, 3, 1, 1, 1, 1, c125_bin, true);
    tk::dnn::Activation a125(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c126(&net, 256, 1, 1, 1, 1, 0, 0, c126_bin, true);
    tk::dnn::Activation a126(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c127(&net, 128, 1, 1, 1, 1, 0, 0, c127_bin, true);
    tk::dnn::Activation a127(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Upsample u128(&net, 2);
    tk::dnn::Layer *r129_layers[1] = {&a54};
    tk::dnn::Route r129(&net, r129_layers, 1);
    tk::dnn::Conv2d c130(&net, 128, 1, 1, 1, 1, 0, 0, c130_bin, true);
    tk::dnn::Activation a130(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer *r131_layers[2] = {&a130,&u128};
    tk::dnn::Route r131(&net, r131_layers, 2);


    tk::dnn::Conv2d c132(&net, 128, 1, 1, 1, 1, 0, 0, c132_bin, true);
    tk::dnn::Activation a132(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c133(&net, 256, 3, 3, 1, 1, 1, 1, c133_bin, true);
    tk::dnn::Activation a133(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c134(&net, 128, 1, 1, 1, 1, 0, 0, c134_bin, true);
    tk::dnn::Activation a134(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c135(&net, 256, 3, 3, 1, 1, 1, 1, c135_bin, true);
    tk::dnn::Activation a135(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c136(&net, 128, 1, 1, 1, 1, 0, 0, c136_bin, true);
    tk::dnn::Activation a136(&net, tk::dnn::ACTIVATION_LEAKY);


    tk::dnn::Conv2d c137(&net, 256, 3, 3, 1, 1, 1, 1, c137_bin, true);
    tk::dnn::Activation a137(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c138(&net, 255, 1, 1, 1, 1, 0, 0, c138_bin, false);
    tk::dnn::Yolo yolo139(&net, classes, 3, g139_bin, 3, 1.2);

    tk::dnn::Layer *r140_layers[1] = {&a136};
    tk::dnn::Route r140(&net, r140_layers, 1);
    tk::dnn::Conv2d c141(&net, 256, 3, 3, 2, 2, 1, 1, c141_bin, true);
    tk::dnn::Activation a141(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer *r142_layers[2] = {&a141,&a126};
    tk::dnn::Route r142(&net, r142_layers, 2);

    tk::dnn::Conv2d c143(&net, 256, 1, 1, 1, 1, 0, 0, c143_bin, true);
    tk::dnn::Activation a143(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c144(&net, 512, 3, 3, 1, 1, 1, 1, c144_bin, true);
    tk::dnn::Activation a144(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c145(&net, 256, 1, 1, 1, 1, 0, 0, c145_bin, true);
    tk::dnn::Activation a145(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c146(&net, 512, 3, 3, 1, 1, 1, 1, c146_bin, true);
    tk::dnn::Activation a146(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c147(&net, 256, 1, 1, 1, 1, 0, 0, c147_bin, true);
    tk::dnn::Activation a147(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Conv2d c148(&net, 512, 3, 3, 1, 1, 1, 1, c148_bin, true);
    tk::dnn::Activation a148(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c149(&net, 255, 1, 1, 1, 1, 0, 0, c149_bin, false);
    tk::dnn::Yolo yolo150(&net, classes, 3, g150_bin, 3, 1.1);

    tk::dnn::Layer *r151_layers[1] = {&a147};
    tk::dnn::Route r151(&net, r151_layers, 1);
    tk::dnn::Conv2d c152(&net, 512, 3, 3, 2, 2, 1, 1, c152_bin, true);
    tk::dnn::Activation a152(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Layer *r153_layers[2] = {&a152,&a116};
    tk::dnn::Route r153(&net, r153_layers, 2);

    tk::dnn::Conv2d c154(&net, 512, 1, 1, 1, 1, 0, 0, c154_bin, true);
    tk::dnn::Activation a154(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c155(&net, 1024, 3, 3, 1, 1, 1, 1, c155_bin, true);
    tk::dnn::Activation a155(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c156(&net, 512, 1, 1, 1, 1, 0, 0, c156_bin, true);
    tk::dnn::Activation a156(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c157(&net, 1024, 3, 3, 1, 1, 1, 1, c157_bin, true);
    tk::dnn::Activation a157(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c158(&net, 512, 1, 1, 1, 1, 0, 0, c158_bin, true);
    tk::dnn::Activation a158(&net, tk::dnn::ACTIVATION_LEAKY);

    tk::dnn::Conv2d c159(&net, 1024, 3, 3, 1, 1, 1, 1, c159_bin, true);
    tk::dnn::Activation a159(&net, tk::dnn::ACTIVATION_LEAKY);
    tk::dnn::Conv2d c160(&net, 255, 1, 1, 1, 1, 0, 0, c160_bin, false);
    tk::dnn::Yolo yolo161(&net, classes, 3, g161_bin, 3, 1.05);


    

    

    yolo[0] = &yolo139;
    yolo[1] = &yolo150;
    yolo[2] = &yolo161;

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
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("yolo4_608"));

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

    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0; 
    for (int i = 0; i < 3; i++)
    {
        printCenteredTitle((std::string(" YOLO ") + std::to_string(i) + " CHECK RESULTS ").c_str(), '=', 30);
        dnnType *out, *out_h;
        int odim = out_dim[i].tot();
        readBinaryFile(output_bins[i], odim, &out_h, &out);
        std::cout<<"CUDNN vs correct"; 
        ret_cudnn |= checkResult(odim, cudnn_out[i], out) == 0 ? 0: ERROR_CUDNN;
        std::cout<<"TRT   vs correct"; 
        ret_tensorrt |= checkResult(odim, rt_out[i], out) == 0 ? 0 : ERROR_TENSORRT;
        std::cout<<"CUDNN vs TRT    "; 
        ret_cudnn_tensorrt |= checkResult(odim, cudnn_out[i], rt_out[i]) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
    }
    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}
