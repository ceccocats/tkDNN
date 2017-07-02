# tkDNN
tkDNN is a Deep Neural Network library built with cuDNN primitives specifically thought to work on NVIDIA TK1 board.<br>
The main scope is to do high performance inference on already trained models.
Currently supports the following layers:

* Dense, fully interconnected
* Activation (RELU, ELU, SIGMOID, TANH)
* Convolutional 2D
* Convolutional 3D
* Max and Average Pooling
* Flatten 
* Data preprocessing

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

## Test
There is a ready to use example on *test* directory, to try it you must generate the weights with Keras
```
cd tests
python test_model.py
```
And then execute the inference on build directory
```
cd build
./tkDNNtest 
```
this should output the same prediction as Keras.

## Simple example
Here is a example of the entire workflow on a simple model.
Using the following Keras model save it to a file
```python
model = Sequential()
model.add(Reshape((20, 1), input_shape=(20)))
model.add(Dense(256))
model.compile()

# save model
model.save("path/to/model.h5")
```

After the model is created the weights can be exported for tkDNN inference
```
python weights_exporter model.h5 dense --output=weights/path
```
the exporter take as arguments, in order:
* input model
* layer type ["dense", "conv2d", conv3d"]
* { layer type ["dense", "conv2d", conv3d"] for each layer to export }
* optional argument --output define path where export weights   

Then we can create a c++ program to do inference on tk1
```c++
#include<tkdnn.h>   //library include

//Network object 
tkDNN::Network net;
//input dimension
tkDNN::dataDim_t dim(1, 20, 1, 1, 1);
//Dense layer
tkDNN::Dense d0(&net, dim, 256, "weights/path", "bias/path");

//here load the input data to CUDA
//value_type is an alias of "float"
value_type *data_d = [...] 

//do inference
value_type *output_d = d0.infer(dim, data_d);
//dim will be updated with the output dimension
``` 
The result is finally stored on output_d in device memory.