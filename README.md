# tkDNN
tkDNN is a Deep Neural Network library built with cuDNN primitives specifically thought to work on NVIDIA TK1 board.<br>
The main scope is to do high performance inference on already trained models.

this branch is actually work on every NVIDIA GPU that support the dependencies:
* CUDA 8
* CUDNN 6
* TENSORRT 2

## Workflow
The recommended workflow follow these step:
* Build and train a model in Keras (on any PC)
* Export weights and bias 
* Define the model on tkDNN
* Do inference (on TK1)

## Compile the library
Build with cmake
```
mkdir build
cd build
cmake ..
make
```
during the cmake configuration it will be dowloaded the weights needed for running
the tests

## Test
Assumiung you have correctly builded the library these are the test ready to exec:
* test_simple: a simple convolutional and dense network (CUDNN only)
* test_mnist: the famous mnist netwok (CUDNN and TENSORRT)
* test_mnistRT: the mnist network hardcoded in using tensorRT apis (TENSORRT only)
* test_yolo: YOLO detection network (CUDNN and TENSORRT)
* test_yolo_tiny: smaller version of YOLO (CUDNN and TENSRRT)

