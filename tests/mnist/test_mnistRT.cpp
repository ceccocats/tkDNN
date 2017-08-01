#include<iostream>
#include<cassert>
#include "tkdnn.h"
#include "NvInfer.h"

const char *input_bin   = "../tests/mnist/input.bin";
const char *c0_bin      = "../tests/mnist/layers/c0.bin";
const char *c1_bin      = "../tests/mnist/layers/c1.bin";
const char *d2_bin      = "../tests/mnist/layers/d2.bin";
const char *d3_bin      = "../tests/mnist/layers/d3.bin";
const char *output_bin   = "../tests/mnist/output.bin";

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

    std::cout<<"\n==== CUDNN ====\n";
    // Network layout
    tkDNN::Network net;
    tkDNN::dataDim_t dim(1, 1, 28, 28, 1);
    tkDNN::Layer *l;
    l = new tkDNN::Conv2d     (&net, dim, 20, 5, 5, 1, 1, 0, 0, c0_bin);
    l = new tkDNN::Pooling    (&net, l->output_dim, 2, 2, 2, 2, tkDNN::POOLING_MAX);
    l = new tkDNN::Conv2d     (&net, l->output_dim, 50, 5, 5, 1, 1, 0, 0, c1_bin);
    l = new tkDNN::Pooling    (&net, l->output_dim, 2, 2, 2, 2, tkDNN::POOLING_MAX);
    l = new tkDNN::Dense      (&net, l->output_dim, 500, d2_bin);
    l = new tkDNN::Activation (&net, l->output_dim, CUDNN_ACTIVATION_RELU);
    l = new tkDNN::Dense      (&net, l->output_dim, 10, d3_bin);
    l = new tkDNN::Softmax    (&net, l->output_dim);
 
    // Load input
    value_type *data;
    value_type *input_h;
    readBinaryFile(input_bin, dim.tot(), &input_h, &data);

    dim.print(); //print initial dimension
    
    TIMER_START
    // Inference
    data = net.infer(dim, data);
    TIMER_STOP
    dim.print();   

    // Print real test
    std::cout<<"\n==== CHECK CUDNN RESULT ====\n";
    value_type *out;
    value_type *out_h;
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

    tkDNN::Conv2d *c0 = (tkDNN::Conv2d*) (net.layers[0]); 
    Weights w { dt, c0->data_h, c0->inputs*c0->outputs*c0->kernelH*c0->kernelW};
    Weights b { dt, c0->bias_h, c0->outputs};
	// Add a convolution layer with 20 outputs and a 5x5 filter.
	auto conv1 = network->addConvolution(*input, 20, DimsHW{5, 5}, w, b);
	assert(conv1 != nullptr);
	conv1->setStride(DimsHW{1, 1});
    conv1->getOutput(0)->setName("out");

/*
	// Add a max pooling layer with stride of 2x2 and kernel size of 2x2.
	auto pool1 = network->addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
	assert(pool1 != nullptr);
	pool1->setStride(DimsHW{2, 2});

	// Add a second convolution layer with 50 outputs and a 5x5 filter.
	auto conv2 = network->addConvolution(*pool1->getOutput(0), 50, DimsHW{5, 5}, weightMap["conv2filter"], weightMap["conv2bias"]);
	assert(conv2 != nullptr);
	conv2->setStride(DimsHW{1, 1});

	// Add a second max pooling layer with stride of 2x2 and kernel size of 2x3>
	auto pool2 = network->addPooling(*conv2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
	assert(pool2 != nullptr);
	pool2->setStride(DimsHW{2, 2});

	// Add a fully connected layer with 500 outputs.
	auto ip1 = network->addFullyConnected(*pool2->getOutput(0), 500, weightMap["ip1filter"], weightMap["ip1bias"]);
	assert(ip1 != nullptr);

	// Add an activation layer using the ReLU algorithm.
	auto relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
	assert(relu1 != nullptr);

	// Add a second fully connected layer with 20 outputs.
	auto ip2 = network->addFullyConnected(*relu1->getOutput(0), OUTPUT_SIZE, weightMap["ip2filter"], weightMap["ip2bias"]);
	assert(ip2 != nullptr);

	// Add a softmax layer to determine the probability.
	auto prob = network->addSoftMax(*ip2->getOutput(0));
	assert(prob != nullptr);
	prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
*/
	network->markOutput(*conv1->getOutput(0));

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

    float output[5*5*20];
	// create GPU buffers and a stream
	checkCuda(cudaMalloc(&buffers[inputIndex], 28*28*sizeof(float)));
	checkCuda(cudaMalloc(&buffers[outputIndex], 5*5*20*sizeof(float)));

	cudaStream_t stream;
	checkCuda(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	checkCuda(cudaMemcpyAsync(buffers[inputIndex], input_h, 1 * 28*28* sizeof(float), cudaMemcpyHostToDevice, stream));
	context->enqueue(1, buffers, stream, nullptr);
	checkCuda(cudaMemcpyAsync(output, buffers[outputIndex],5*5*20*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);


    std::cout<<"\n==== CHECK CUDNN RESULT ====\n";
    std::cout<<"Diff: "<<checkResult(dim.tot(), (float*)buffers[outputIndex], c0->dstData)<<"\n";

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	checkCuda(cudaFree(buffers[inputIndex]));
	checkCuda(cudaFree(buffers[outputIndex]));

	// destroy the engine
	context->destroy();
	engine->destroy();

    return 0;
}
