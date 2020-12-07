#include<iostream>
#include<cassert>
#include "tkdnn.h"
#include "NvInfer.h"

const char *input_bin   = "mnist/input.bin";
const char *c0_bin      = "mnist/layers/c0.bin";
const char *c1_bin      = "mnist/layers/c1.bin";
const char *d2_bin      = "mnist/layers/d2.bin";
const char *d3_bin      = "mnist/layers/d3.bin";
const char *output_bin   = "mnist/output.bin";

using namespace nvinfer1;

// Logger for info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

int main() {

	downloadWeightsifDoNotExist(input_bin, "mnist", "https://cloud.hipert.unimore.it/s/2TyQkMJL3LArLAS/download");

    std::cout<<"\n==== CUDNN ====\n";
    // Network layout
    tk::dnn::dataDim_t dim(1, 1, 28, 28, 1);
	tk::dnn::Network net(dim);
    tk::dnn::Conv2d     l0(&net, 20, 5, 5, 1, 1, 0, 0, c0_bin);
    tk::dnn::Pooling    l1(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
    tk::dnn::Conv2d     l2(&net, 50, 5, 5, 1, 1, 0, 0, c1_bin);
    tk::dnn::Pooling    l3(&net, 2, 2, 2, 2, 0, 0, tk::dnn::POOLING_MAX);
    tk::dnn::Dense      l4(&net, 500, d2_bin);
    tk::dnn::Activation l5(&net, CUDNN_ACTIVATION_RELU);
    tk::dnn::Dense      l6(&net, 10, d3_bin);
    tk::dnn::Softmax    l7(&net);
 
    // Load input
    dnnType *data;
    dnnType *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dim.print(); //print initial dimension
    
    // Inference
    {
        TKDNN_TSTART
        data = net.infer(dim, data);
        TKDNN_TSTOP
        dim.print();   
    }

    // Print real test
    std::cout<<"\n==== CHECK CUDNN RESULT ====\n";
    dnnType *out;
    dnnType *out_h;
    readBinaryFile(output_bin, dim.tot(), &out_h, &out);
    std::cout<<"Diff: "<<checkResult(dim.tot(), out, data)<<"\n";
 

    std::cout<<"\n==== TensorRT ====\n";
	// create the builder
	IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	INetworkDefinition* network = builder->createNetwork();

    DataType dt = DataType::kFLOAT;
	//  Create input of shape { 1, 1, 28, 28 } with name referenced by "data"
	auto input = network->addInput("data", dt, DimsCHW{ 1, 28, 28});
	assert(input != nullptr);

    tk::dnn::Conv2d *c0 = &l0; 
    Weights w { dt, c0->data_h, c0->inputs*c0->outputs*c0->kernelH*c0->kernelW};
    Weights b { dt, c0->bias_h, c0->outputs};
	// Add a convolution layer with 20 outputs and a 5x5 filter.
	auto conv1 = network->addConvolution(*input, 20, DimsHW{5, 5}, w, b);
	assert(conv1 != nullptr);
	conv1->setStride(DimsHW{1, 1});

	// Add a max pooling layer with stride of 2x2 and kernel size of 2x2.
	auto pool1 = network->addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
	assert(pool1 != nullptr);
	pool1->setStride(DimsHW{2, 2});

    tk::dnn::Conv2d *c1 = &l2; 
    Weights w1 { dt, c1->data_h, c1->inputs*c1->outputs*c1->kernelH*c1->kernelW};
    Weights b1 { dt, c1->bias_h, c1->outputs};
	// Add a second convolution layer with 50 outputs and a 5x5 filter.
	auto conv2 = network->addConvolution(*pool1->getOutput(0), 50, DimsHW{5, 5}, w1, b1);
	assert(conv2 != nullptr);
	conv2->setStride(DimsHW{1, 1});

	// Add a second max pooling layer with stride of 2x2 and kernel size of 2x3>
	auto pool2 = network->addPooling(*conv2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
	assert(pool2 != nullptr);
	pool2->setStride(DimsHW{2, 2});

    tk::dnn::Dense *d2 = &l4; 
    Weights w2 { dt, d2->data_h, d2->inputs*d2->outputs};
    Weights b2 { dt, d2->bias_h, d2->outputs};
	// Add a fully connected layer with 500 outputs.
	auto ip1 = network->addFullyConnected(*pool2->getOutput(0), 500, w2, b2);
	assert(ip1 != nullptr);

	// Add an activation layer using the ReLU algorithm.
	auto relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
	assert(relu1 != nullptr);

    tk::dnn::Dense *d3 = &l6; 
    Weights w3 { dt, d3->data_h, d3->inputs*d3->outputs};
    Weights b3 { dt, d3->bias_h, d3->outputs};
	// Add a second fully connected layer with 20 outputs.
	auto ip2 = network->addFullyConnected(*relu1->getOutput(0), 10, w3, b3);
	assert(ip2 != nullptr);

	// Add a softmax layer to determine the probability.
	auto prob = network->addSoftMax(*ip2->getOutput(0));
	assert(prob != nullptr);
	prob->getOutput(0)->setName("out");

	network->markOutput(*prob->getOutput(0));

	// Build the engine
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 20);

	auto engine = builder->buildCudaEngine(*network);
	// we don't need the network any more
	network->destroy();

	IExecutionContext *context = engine->createExecutionContext();

	// run inference
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine->getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine->getBindingIndex("data"); 
    int outputIndex = engine->getBindingIndex("out");

    float output[10];
	// create GPU buffers and a stream
	checkCuda(cudaMalloc(&buffers[inputIndex], 28*28*sizeof(float)));
	checkCuda(cudaMalloc(&buffers[outputIndex], 10*sizeof(float)));

	cudaStream_t stream;
	checkCuda(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    {
        checkCuda(cudaMemcpyAsync(buffers[inputIndex], input_h, 1 * 28*28* sizeof(float), cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);  //want to test only the inference time
        TKDNN_TSTART
        context->enqueue(1, buffers, stream, nullptr);
        TKDNN_TSTOP
        checkCuda(cudaMemcpyAsync(output, buffers[outputIndex],10*sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    std::cout<<"\n==== CHECK CUDNN RESULT ====\n";
    std::cout<<"Diff: "<<checkResult(dim.tot(), (float*)buffers[outputIndex], data)<<"\n";

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	checkCuda(cudaFree(buffers[inputIndex]));
	checkCuda(cudaFree(buffers[outputIndex]));

	// destroy the engine
	context->destroy();
	engine->destroy();

    return 0;
}
