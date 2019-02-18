# tkDNN
tkDNN is a Deep Neural Network library built with cuDNN primitives specifically thought to work on NVIDIA TK1(and all successive) board.<br>
The main scope is to do high performance inference on already trained models.

this branch actually work on every NVIDIA GPU that support the dependencies:
* CUDA 9
* CUDNN 7.105
* TENSORRT 4.02

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
* test_yolo3_berkeley: our yolo3 version trained with BDD100K dateset 

## yolo3 berkeley demo detection
For the live detection you need to precompile the tensorRT file by luncing the desidered network test, this is the recommended process:
```
export TKDNN_MODE=FP16   # set the half floating point optimization
rm yolo3_berkeley.rt		 # be sure to delete(or move) old tensorRT files
./test_yolo3_berkeley              # run the yolo test (is slow)
# with f16 inference the result will be a bit incorrect
```
this will genereate a yolo3_berkeley.rt file that can be used for live detection:
```
./demo             # launch detection on a demo video
./demo /dev/video0 # launch detection on device 0
```


