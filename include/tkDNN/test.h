
#include <tkdnn.h>
int testInference(std::vector<std::string> input_bins, std::vector<std::string> output_bins, 
    tk::dnn::Network *net, tk::dnn::NetworkRT *netRT = nullptr) {

    std::vector<tk::dnn::Layer*> outputs;
    for(int i=0; i<net->num_layers; i++) {
        if(net->layers[i]->final)
            outputs.push_back(net->layers[i]);
    }
    // no final layers, set last as output
    if(outputs.size() == 0) {
        outputs.push_back(net->layers[net->num_layers-1]);
    }


    // check input
    if(input_bins.size() != 1) {
        FatalError("currently support only 1 input");
    }
    if(output_bins.size() != outputs.size()) {
        std::cout<<output_bins.size()<<" "<<outputs.size()<<"\n";
        FatalError("outputs size mismatch");
    }

    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bins[0], net->input_dim.tot(), &input_h, &data);

    // outputs
    //dnnType *cudnn_out[outputs.size()], *rt_out[outputs.size()];
    std::vector<dnnType *> cudnn_out,rt_out;

    tk::dnn::dataDim_t dim1 =  net->input_dim; //input dim
    printCenteredTitle(" CUDNN inference ", '=', 30); {
        dim1.print();
        TKDNN_TSTART
        net->infer(dim1, data);    
        TKDNN_TSTOP
        dim1.print();   
    }
    for(int i=0; i<outputs.size(); i++) cudnn_out.push_back(outputs[i]->dstData);

    if(netRT != nullptr) {
        tk::dnn::dataDim_t dim2 = net->input_dim;
        printCenteredTitle(" TENSORRT inference ", '=', 30); {
            dim2.print();
            TKDNN_TSTART
            netRT->infer(dim2, data);
            TKDNN_TSTOP
            dim2.print();
        }
        for(int i=0; i<outputs.size(); i++) rt_out.push_back((dnnType*)netRT->buffersRT[i+1]);
    }

    int ret_cudnn = 0, ret_tensorrt = 0, ret_cudnn_tensorrt = 0; 
    for(int i=0; i<outputs.size(); i++) {
        printCenteredTitle((std::string(" OUTPUT ") + std::to_string(i) + " CHECK RESULTS ").c_str(), '=', 30);
        dnnType *out, *out_h;
        int odim = outputs[i]->output_dim.tot();
        readBinaryFile(output_bins[i], odim, &out_h, &out);
        std::cout<<"CUDNN vs correct"; 
        ret_cudnn |= checkResult(odim, cudnn_out[i], out) == 0 ? 0: ERROR_CUDNN;
        if(netRT != nullptr) {
            std::cout<<"TRT   vs correct"; 
            ret_tensorrt |= checkResult(odim, rt_out[i], out) == 0 ? 0 : ERROR_TENSORRT;
            std::cout<<"CUDNN vs TRT    "; 
            ret_cudnn_tensorrt |= checkResult(odim, cudnn_out[i], rt_out[i]) == 0 ? 0 : ERROR_CUDNNvsTENSORRT;
        }

        delete [] out_h;
        checkCuda( cudaFree(out) );
    }
    delete [] input_h;
    checkCuda( cudaFree(data) );
    return ret_cudnn | ret_tensorrt | ret_cudnn_tensorrt;
}