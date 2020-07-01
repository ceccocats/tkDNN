# tkDNN
tkDNN is a Deep Neural Network library built with cuDNN and tensorRT primitives, specifically thought to work on NVIDIA Jetson Boards. It has been tested on TK1(branch cudnn2), TX1, TX2, AGX Xavier and several discrete GPU.
The main goal of this project is to exploit NVIDIA boards as much as possible to obtain the best inference performance. It does not allow training. 


If you use tkDNN in your research, please cite one of the following papers. For use in commercial solutions, write at gattifrancesco@hotmail.it or refer to https://hipert.unimore.it/ .

```
Accepted paper @ IRC 2020, will soon be published.
M. Verucchi, L. Bartoli, F. Bagni, F. Gatti, P. Burgio and M. Bertogna, "Real-Time clustering and LiDAR-camera fusion on embedded platforms for self-driving cars",  in proceedings in IEEE Robotic Computing (2020)

Accepted paper @ ETFA 2020, will soon be published.
M. Verucchi, G. Brilli, D. Sapienza, M. Verasani, M. Arena, F. Gatti, A. Capotondi, R. Cavicchioli, M. Bertogna, M. Solieri
"A Systematic Assessment of Embedded Neural Networks for Object Detection", in IEEE International Conference on Emerging Technologies and Factory Automation (2020)
```

## Results
Inference FPS of yolov4 with tkDNN, average of 1200 images with the same dimesion as the input size, on 
  * RTX 2080Ti (CUDA 10.2, TensorRT 7.0.0, Cudnn 7.6.5);
  * Xavier AGX, Jetpack 4.3 (CUDA 10.0, CUDNN 7.6.3, tensorrt 6.0.1 );
  * Tx2, Jetpack 4.2 (CUDA 10.0, CUDNN 7.3.1, tensorrt 5.0.6 );
  * Jetson Nano, Jetpack 4.4  (CUDA 10.2, CUDNN 8.0.0, tensorrt 7.1.0 ). 

| Platform   | Network    | FP32, B=1 | FP32, B=4	| FP16, B=1 |	FP16, B=4 |	INT8, B=1 |	INT8, B=4 | 
| :------:   | :-----:    | :-----:   | :-----:   | :-----:   |	:-----:   |	:-----:   |	:-----:   | 
| RTX 2080Ti | yolo4  320 | 118,59	  |237,31	    | 207,81	  | 443,32	  | 262,37	  | 530,93    | 
| RTX 2080Ti | yolo4  416 | 104,81	  |162,86	    | 169,06	  | 293,78	  | 206,93	  | 353,26    | 
| RTX 2080Ti | yolo4  512 | 92,98	    |132,43	    | 140,36	  | 215,17	  | 165,35	  | 254,96    | 
| RTX 2080Ti | yolo4  608 | 63,77	    |81,53	    | 111,39	  | 152,89	  | 127,79	  | 184,72    | 
| AGX Xavier | yolo4 320  |	26,78	    |32,05	    | 57,14	    | 79,05	    | 73,15	    | 97,56     |
| AGX Xavier | yolo4 416  |	19,96	    |21,52	    | 41,01	    | 49,00	    | 50,81	    | 60,61     |
| AGX Xavier | yolo4 512  |	16,58	    |16,98	    | 31,12	    | 33,84	    | 37,82	    | 41,28     |
| AGX Xavier | yolo4 608  |	9,45 	    |10,13	    | 21,92	    | 23,36	    | 27,05	    | 28,93     |
| Tx2        | yolo4 320	| 11,18	    | 12,07	    | 15,32	    | 16,31     | -         | -         |
| Tx2        | yolo4 416	| 7,30	    | 7,58	    | 9,45	    | 9,90      | -         | -         |
| Tx2        | yolo4 512	| 5,96	    | 5,95	    | 7,22	    | 7,23      | -         | -         |
| Tx2        | yolo4 608	| 3,63	    | 3,65	    | 4,67	    | 4,70      | -         | -         |
| Nano       | yolo4 320	| 4,23	    | 4,55	    | 6,14	    | 6,53      | -         | -         |
| Nano       | yolo4 416	| 2,88	    | 3,00	    | 3,90	    | 4,04      | -         | -         |
| Nano       | yolo4 512	| 2,32	    | 2,34	    | 3,02	    | 3,04      | -         | -         |
| Nano       | yolo4 608	| 1,40	    | 1,41	    | 1,92	    | 1,93      | -         | -         |

## Index
- [tkDNN](#tkdnn)
  - [Index](#index)
  - [Dependencies](#dependencies)
  - [About OpenCV](#about-opencv)
  - [How to compile this repo](#how-to-compile-this-repo)
  - [Workflow](#workflow)
  - [How to export weights](#how-to-export-weights)
    - [1)Export weights from darknet](#1export-weights-from-darknet)
    - [2)Export weights for DLA34 and ResNet101](#2export-weights-for-dla34-and-resnet101)
    - [3)Export weights for CenterNet](#3export-weights-for-centernet)
    - [4)Export weights for MobileNetSSD](#4export-weights-for-mobilenetssd)
  - [Run the demo](#run-the-demo)
    - [FP16 inference](#fp16-inference)
    - [INT8 inference](#int8-inference)
  - [mAP demo](#map-demo)
  - [Existing tests and supported networks](#existing-tests-and-supported-networks)
  - [References](#references)




## Dependencies
This branch works on every NVIDIA GPU that supports the dependencies:
* CUDA 10.0
* CUDNN 7.603
* TENSORRT 6.01
* OPENCV 3.4
* yaml-cpp 0.5.2 (sudo apt install libyaml-cpp-dev)

## About OpenCV
To compile and install OpenCV4 with contrib us the script ```install_OpenCV4.sh```. It will download and compile OpenCV in Download folder.
```
bash scripts/install_OpenCV4.sh
```
When using openCV not compiled with contrib, comment the definition of OPENCV_CUDACONTRIBCONTRIB in include/tkDNN/DetectionNN.h. When commented, the preprocessing of the networks is computed on the CPU, otherwise on the GPU. In the latter case some milliseconds are saved in the end-to-end latency. 

## How to compile this repo
Build with cmake. If using Ubuntu 18.04 a new version of cmake is needed (3.15 or above). 
```
git clone https://github.com/ceccocats/tkDNN
cd tkDNN
mkdir build
cd build
cmake .. 
make
```

## Workflow
Steps needed to do inference on tkDNN with a custom neural network. 
* Build and train a NN model with your favorite framework.
* Export weights and bias for each layer and save them in a binary file (one for layer).
* Export outputs for each layer and save them in a binary file (one for layer).
* Create a new test and define the network, layer by layer using the weights extracted and the output to check the results. 
* Do inference.

## How to export weights

Weights are essential for any network to run inference. For each test a folder organized as follow is needed (in the build folder):
```
    test_nn
        |---- layers/ (folder containing a binary file for each layer with the corresponding wieghts and bias)
        |---- debug/  (folder containing a binary file for each layer with the corresponding outputs)
```
Therefore, once the weights have been exported, the folders layers and debug should be placed in the corresponding test.

### 1)Export weights from darknet
To export weights for NNs that are defined in darknet framework, use [this](https://git.hipert.unimore.it/fgatti/darknet.git) fork of darknet and follow these steps to obtain a correct debug and layers folder, ready for tkDNN.

```
git clone https://git.hipert.unimore.it/fgatti/darknet.git
cd darknet
make
mkdir layers debug
./darknet export <path-to-cfg-file> <path-to-weights> layers
```
N.b. Use compilation with CPU (leave GPU=0 in Makefile) if you also want debug. 

### 2)Export weights for DLA34 and ResNet101 
To get weights and outputs needed to run the tests dla34 and resnet101 use the Python script and the Anaconda environment included in the repository.   

Create Anaconda environment and activate it:
```
conda env create -f file_name.yml
source activate env_name
python <script name>
```
### 3)Export weights for CenterNet
To get the weights needed to run Centernet tests use [this](https://github.com/sapienzadavide/CenterNet.git) fork of the original Centernet. 
```
git clone https://github.com/sapienzadavide/CenterNet.git
```
* follow the instruction in the README.md and INSTALL.md

```
python demo.py --input_res 512 --arch resdcn_101 ctdet --demo /path/to/image/or/folder/or/video/or/webcam --load_model ../models/ctdet_coco_resdcn101.pth --exp_wo --exp_wo_dim 512
python demo.py --input_res 512 --arch dla_34 ctdet --demo /path/to/image/or/folder/or/video/or/webcam --load_model ../models/ctdet_coco_dla_2x.pth --exp_wo --exp_wo_dim 512
```
### 4)Export weights for MobileNetSSD
To get the weights needed to run Mobilenet tests use [this](https://github.com/mive93/pytorch-ssd) fork of a Pytorch implementation of SSD network. 

```
git clone https://github.com/mive93/pytorch-ssd
cd pytorch-ssd
conda env create -f env_mobv2ssd.yml
python run_ssd_live_demo.py mb2-ssd-lite <pth-model-fil> <labels-file>
```

## Darknet Parser
tkDNN implement and easy parser for darknet cfg files, a network can be converted with *tk::dnn::darknetParser*:
```
// example of parsing yolo4
tk::dnn::Network *net = tk::dnn::darknetParser("yolov4.cfg", "yolov4/layers", "coco.names");
net->print();
```
All models from darknet are now parsed directly from cfg, you still need to export the weights with the descripted tools in the previus section.
<details>
  <summary>Supported layers</summary>
  convolutional
  maxpool
  avgpool
  shortcut
  upsample
  route
  reorg
  region
  yolo
</details>
<details>
  <summary>Supported activations</summary>
  relu
  leaky
  mish
</details>

## Run the demo

To run the an object detection demo follow these steps (example with yolov3):
```
rm yolo3_fp32.rt        # be sure to delete(or move) old tensorRT files
./test_yolo3            # run the yolo test (is slow)
./demo yolo3_fp32.rt ../demo/yolo_test.mp4 y
```
In general the demo program takes 4 parameters:
```
./demo <network-rt-file> <path-to-video> <kind-of-network> <number-of-classes> <n-batches> <show-flag>
```
where
*  ```<network-rt-file>``` is the rt file generated by a test
*  ```<<path-to-video>``` is the path to a video file or a camera input  
*  ```<kind-of-network>``` is the type of network. Thee types are currently supported: ```y``` (YOLO family), ```c``` (CenterNet family) and ```m``` (MobileNet-SSD family)
*  ```<number-of-classes>```is the number of classes the network is trained on
*  ```<n-batches>``` number of batches to use in inference (N.B. you should first export TKDNN_BATCHSIZE to the required n_batches and create again the rt file for the network).
*  ```<show-flag>``` if set to 0 the demo will not show the visualization but save the video into result.mp4 (if n-batches ==1)

N.b. By default it is used FP32 inference

![demo](https://user-images.githubusercontent.com/11562617/72547657-540e7800-388d-11ea-83c6-49dfea2a0607.gif)

### FP16 inference

To run the an object detection demo with FP16 inference follow these steps (example with yolov3):
```
export TKDNN_MODE=FP16  # set the half floating point optimization
rm yolo3_fp16.rt        # be sure to delete(or move) old tensorRT files
./test_yolo3            # run the yolo test (is slow)
./demo yolo3_fp16.rt ../demo/yolo_test.mp4 y
```
N.b. Using FP16 inference will lead to some errors in the results (first or second decimal). 

### INT8 inference

To run the an object detection demo with INT8 inference three environment variables need to be set:
  * ```export TKDNN_MODE=INT8```: set the 8-bit integer optimization
  * ```export TKDNN_CALIB_IMG_PATH=/path/to/calibration/image_list.txt``` : image_list.txt has in each line the absolute path to a calibration image
  * ```export TKDNN_CALIB_LABEL_PATH=/path/to/calibration/label_list.txt```: label_list.txt has in each line the absolute path to a calibration label
  
You should provide image_list.txt and label_list.txt, using training images. However, if you want to quickly test the INT8 inference you can run (from this repo root folder)
```
bash scripts/download_validation.sh COCO
```
to automatically download COCO2017 validation (inside demo folder) and create those needed file. Use BDD insted of COCO to download BDD validation. 

Then a complete example using yolo3 and COCO dataset would be:
```
export TKDNN_MODE=INT8
export TKDNN_CALIB_LABEL_PATH=../demo/COCO_val2017/all_labels.txt
export TKDNN_CALIB_IMG_PATH=../demo/COCO_val2017/all_images.txt
rm yolo3_int8.rt        # be sure to delete(or move) old tensorRT files
./test_yolo3            # run the yolo test (is slow)
./demo yolo3_int8.rt ../demo/yolo_test.mp4 y
```
N.B. 
 * Using INT8 inference will lead to some errors in the results. 
 * The test will be slower: this is due to the INT8 calibration, which may take some time to complete. 
 * INT8 calibration requires TensorRT version greater than or equal to 6.0
 * Only 100 images are used to create the calibration table by default (set in the code).

### BatchSize bigger than 1
```
export TKDNN_BATCHSIZE=2
# build tensorRT files
```
This will create a TensorRT file with the desidered **max** batch size.
The test will still run with a batch of 1, but the created tensorRT can manage the desidered batch size.

### Test batch Inference
This will test the network with random input and check if the output of each batch is the same.
```
./test_rtinference <network-rt-file> <number-of-batches>
# <number-of-batches> should be less or equal to the max batch size of the <network-rt-file>

# example
export TKDNN_BATCHSIZE=4           # set max batch size
rm yolo3_fp32.rt                   # be sure to delete(or move) old tensorRT files
./test_yolo3                       # build RT file
./test_rtinference yolo3_fp32.rt 4 # test with a batch size of 4
```

## mAP demo

To compute mAP, precision, recall and f1score, run the map_demo.

A validation set is needed. 
To download COCO_val2017 (80 classes) run (form the root folder): 
```
bash scripts/download_validation.sh COCO
```
To download Berkeley_val (10 classes) run (form the root folder): 
```
bash scripts/download_validation.sh BDD
```

To compute the map, the following parameters are needed:
```
./map_demo <network rt> <network type [y|c|m]> <labels file path> <config file path>
```
where 
* ```<network rt>```: rt file of a chosen network on which compute the mAP.
* ```<network type [y|c|m]>```: type of network. Right now only y(yolo), c(centernet) and m(mobilenet) are allowed
* ```<labels file path>```: path to a text file containing all the paths of the ground-truth labels. It is important that all the labels of the ground-truth are in a folder called 'labels'. In the folder containing the folder 'labels' there should be also a folder 'images', containing all the ground-truth images having the same same as the labels. To better understand, if there is a label path/to/labels/000001.txt there should be a corresponding image path/to/images/000001.jpg. 
* ```<config file path>```: path to a yaml file with the parameters needed for the mAP computation, similar to demo/config.yaml

Example:

```
cd build
./map_demo dla34_cnet_FP32.rt c ../demo/COCO_val2017/all_labels.txt ../demo/config.yaml
```

This demo also creates a json file named ```net_name_COCO_res.json``` containing all the detections computed. The detections are in COCO format, the correct format to subit the results to [CodaLab COCO detection challenge](https://competitions.codalab.org/competitions/20794#participate).

## Existing tests and supported networks

| Test Name         | Network                                       | Dataset                                                       | N Classes | Input size    | Weights                                                                   |
| :---------------- | :-------------------------------------------- | :-----------------------------------------------------------: | :-------: | :-----------: | :------------------------------------------------------------------------ |
| yolo              | YOLO v2<sup>1</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 608x608       | [weights](https://cloud.hipert.unimore.it/s/nf4PJ3k8bxBETwL/download)                                                                   |
| yolo_224          | YOLO v2<sup>1</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| yolo_berkeley     | YOLO v2<sup>1</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)   | 10        | 416x736       | weights                                                                   |
| yolo_relu         | YOLO v2 (with ReLU, not Leaky)<sup>1</sup>    | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | weights                                                                   |
| yolo_tiny         | YOLO v2 tiny<sup>1</sup>                      | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/m3orfJr8pGrN5mQ/download)                                                                   |
| yolo_voc          | YOLO v2<sup>1</sup>                           | [VOC      ](http://host.robots.ox.ac.uk/pascal/VOC/)          | 21        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/DJC5Fi2pEjfNDP9/download)                                                                   |
| yolo3             | YOLO v3<sup>2</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/jPXmHyptpLoNdNR/download)     |
| yolo3_512   | YOLO v3<sup>2</sup>                                 | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/RGecMeGLD4cXEWL/download)     |
| yolo3_berkeley    | YOLO v3<sup>2</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)   | 10        | 320x544       | [weights](https://cloud.hipert.unimore.it/s/o5cHa4AjTKS64oD/download)                                                                   |
| yolo3_coco4       | YOLO v3<sup>2</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 4         | 416x416       | [weights](https://cloud.hipert.unimore.it/s/o27NDzSAartbyc4/download)                                                                   |
| yolo3_flir        | YOLO v3<sup>2</sup>                           | [FREE FLIR](https://www.flir.com/oem/adas/adas-dataset-form/) | 3         | 320x544       | [weights](https://cloud.hipert.unimore.it/s/62DECncmF6bMMiH/download)                                                                   |
| yolo3_tiny        | YOLO v3 tiny<sup>2</sup>                      | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/LMcSHtWaLeps8yN/download)     |
| yolo3_tiny512     | YOLO v3 tiny<sup>2</sup>                      | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/8Zt6bHwHADqP4JC/download)     |
| dla34             | Deep Leayer Aggreagtion (DLA) 34<sup>3</sup>  | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| dla34_cnet        | Centernet (DLA34 backend)<sup>4</sup>         | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/KRZBbCQsKAtQwpZ/download)     |
| mobilenetv2ssd    | Mobilnet v2 SSD Lite<sup>5</sup>              | [VOC      ](http://host.robots.ox.ac.uk/pascal/VOC/)          | 21        | 300x300       | [weights](https://cloud.hipert.unimore.it/s/x4ZfxBKN23zAJQp/download)     |
| mobilenetv2ssd512 | Mobilnet v2 SSD Lite<sup>5</sup>              | [COCO 2017](http://cocodataset.org/)                          | 81        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/pdCw2dYyHMJrcEM/download)     |
| resnet101         | Resnet 101<sup>6</sup>                        | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| resnet101_cnet    | Centernet (Resnet101 backend)<sup>4</sup>     | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/5BTjHMWBcJk8g3i/download)     |
| csresnext50-panet-spp    | Cross Stage Partial Network <sup>7</sup>     | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/Kcs4xBozwY4wFx8/download)     |
| yolo4             | Yolov4 <sup>8</sup>                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/d97CFzYqCPCp5Hg/download)     |
| yolo4_berkeley             | Yolov4 <sup>8</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)                          | 10        | 540x320       | [weights](https://cloud.hipert.unimore.it/s/nkWFa5fgb4NTdnB/download)     |
| yolo4tiny             | Yolov4 tiny                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/iRnc4pSqmx78gJs/download)     |


## References

1. Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
3. Yu, Fisher, et al. "Deep layer aggregation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
4. Zhou, Xingyi, Dequan Wang, and Philipp Krähenbühl. "Objects as points." arXiv preprint arXiv:1904.07850 (2019).
5. Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
6. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
7. Wang, Chien-Yao, et al. "CSPNet: A New Backbone that can Enhance Learning Capability of CNN." arXiv preprint arXiv:1911.11929 (2019).
8. Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "YOLOv4: Optimal Speed and Accuracy of Object Detection." arXiv preprint arXiv:2004.10934 (2020).
