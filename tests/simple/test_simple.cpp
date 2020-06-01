#include<iostream>
#include "tkdnn.h"

const char *input_bin   = "simple/input.bin";
const char *c0_bin      = "simple/layers/conv1d_1.bin";
const char *l1_bin      = "simple/layers/bidirectional_1.bin";
const char *l2_bin      = "simple/layers/bidirectional_2.bin";
const char *output_bin  = "simple/output.bin";

int main() {

    // Network layout
    tk::dnn::dataDim_t dim(1, 8, 1, 3);
    tk::dnn::Network net(dim);
    tk::dnn::Conv2d     l0(&net, 4, 1, 2, 1, 1, 0, 0, c0_bin);
    tk::dnn::LSTM       l1(&net, 5, true, l1_bin);
    tk::dnn::LSTM       l2(&net, 5, false, l2_bin);

    net.print();

    net.print();

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    // Print input
    std::cout<<"\n======= INPUT =======\n";
    printDeviceVector(dim.tot(), data);
    std::cout<<"\n";

    //convert network to tensorRT
    tk::dnn::NetworkRT netRT(&net, net.getNetworkRTName("simple"));

    dnnType *out_data, *out_data2; // cudnn output, tensorRT output

    tk::dnn::dataDim_t dim1 = dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TKDNN_TSTART
        out_data = net.infer(dim1, data);
        TKDNN_TSTOP
        dim1.print();
    }

    tk::dnn::dataDim_t dim2 = dim;
    printCenteredTitle(" TENSORRT inference ", '=', 30); {
        dim2.print();
        TKDNN_TSTART
        out_data2 = netRT.infer(dim2, data);
        TKDNN_TSTOP
        dim2.print();
    }

    std::cout<<"\n======= CUDNN =======\n";
    printDeviceVector(dim.tot(), out_data);
    std::cout<<"\n======= TENSORRT =======\n";
    printDeviceVector(dim.tot(), out_data2);

    printCenteredTitle(" CHECK RESULTS ", '=', 30);
    dnnType *out, *out_h;
    int out_dim = net.getOutputDim().tot();
    //readBinaryFile(output_bin, out_dim, &out_h, &out);
    // std::cout<<"CUDNN vs correct"; 
    // int ret_cudnn = checkResult(out_dim, out_data, out) == 0 ? 0: ERROR_CUDNN;
    // std::cout<<"TRT   vs correct"; 
    // int ret_tensorrt = checkResult(out_dim, out_data2, out) == 0 ? 0 : ERROR_TENSORRT;
    std::cout<<"CUDNN vs TRT    "; 
    int ret_cudnn_tensorrt = checkResult(out_dim, out_data, out_data2) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;

    return ret_cudnn_tensorrt;
}
