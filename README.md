# tkDNN
tkDNN is a Deep Neural Network library built with cuDNN primitives specifically thought to work on NVIDIA TK1(and all successive) board.<br>
The main scope is to do high performance inference on already trained models.

this branch actually work on every NVIDIA GPU that support the dependencies:
* CUDA 10.0
* CUDNN 7.603
* TENSORRT 6.01
* OPENCV 4.1
* yaml-cpp 0.5.2 (sudo apt install libyaml-cpp-dev)

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
# use -DTEST_DATA=False to skip dataset download
make
```
during the cmake configuration it will be dowloaded the weights needed for running
the tests

## DLA34 and ResNet101 weights
To get weights and outputs needed for running the tests you can use the Python 
script and the Anaconda environment included in the repository.   

Create Anaconda environment and activate it:
```
conda env create -f file_name.yml
source activate env_name 
```
Run the Python script inside the environment.

## CenterNet weights
To get the weights needed for running the tests:

* clone the forked repository by the original CenterNet:
```
git clone https://github.com/sapienzadavide/CenterNet.git
```
* follow the instruction in the README.md and INSTALL.md
* copy the weigths and outputs from /path/to/CenterNet/src/ in ./test/centernet-path/ . For example:
```
cp /path/to/CenterNet/src/layers_dla/* ./test/dla34_cnet/layers/
cp /path/to/CenterNet/src/debug_dla/* ./test/dla34_cnet/debug/
```
or
```
cp /path/to/CenterNet/src/layers_resdcn/* ./test/resnet101_cnet/layers/
cp /path/to/CenterNet/src/debug_resdcn/* ./test/resnet101_cnet/debug/
```

## Test
Assumiung you have correctly builded the library these are the test ready to exec:
* test_simple: a simple convolutional and dense network (CUDNN only)
* test_mnist: the famous mnist netwok (CUDNN and TENSORRT)
* test_mnistRT: the mnist network hardcoded in using tensorRT apis (TENSORRT only)
* test_yolo: YOLO detection network (CUDNN and TENSORRT)
* test_yolo_tiny: smaller version of YOLO (CUDNN and TENSRRT)
* test_yolo3_berkeley: our yolo3 version trained with BDD100K dateset 
* test_resnet101: ResNet101 network (CUDNN and TENSORRT)
* test_resnet101_cnet: CenterNet detection based on ResNet101 (CUDNN and TENSORRT)
* test_dla34: DLA34 network (CUDNN and TENSORRT)
* test_dla34_cnet: CenterNet detection based on DLA34 (CUDNN and TENSORRT)


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
./demo                                 # launch detection on a demo video
./demo yolo3_berkeley.rt /dev/video0 y # launch detection on device 0
```
![demo](https://user-images.githubusercontent.com/11562617/72547657-540e7800-388d-11ea-83c6-49dfea2a0607.gif)


## CenterNet (DLA34, ResNet101) demo detection
For the live detection you need to precompile the tensorRT file by luncing the desidered network test, this is the recommended process:
```
export TKDNN_MODE=FP16   # set the half floating point optimization
```

For CenterNet based on ResNet101:
```
rm resnet101_cnet.rt		 # be sure to delete(or move) old tensorRT files
./test_resnet101_cnet              # run the yolo test (is slow)
# with f16 inference the result will be a bit incorrect
```

For CenterNet based on DLA34:
```
rm dla34_cnet.rt		     # be sure to delete(or move) old tensorRT files
./test_dla34_cnet                  # run the yolo test (is slow)
# with f16 inference the result will be a bit incorrect
```

this will genereate resnet101_cnet.rt and dla34_cnet.rt file that can be used for live detection:
```
./demo dla34_cnet.rt ../demo/yolo_test.mp4 c    # launch detection on a demo video
./demo resnet101_cnet.rt /dev/video0 c          # launch detection on device 0
./demo dla34_cnet.rt /dev/video0 c              # launch detection on device 0
```

## mAP demo
To compute mAP, precision, recall and f1score, run the map_demo.

A validation set is needed. To download COCO_val2017 run (form the root folder): 
```
bash download_validation.sh 
```

To compute the map, the following parameters are needed:
```
./map_demo <network rt> <network type [y|c]> <labels file path> <config file path>
```
where 
* ```<network rt>```: rt file of a choosen network on wich compute the mAP.
* ```<network type [y|c]>```: type of network. Right now only y(yolo) and c(centernet) are allowed
* ```<labels file path>```: path to a text file containing all the paths of the groundtruth labels. It is important that all the labels of the groundtruth are in a folder called 'labels'. In the folder containing the folder 'labels' there should be also a folder 'images', containing all the groundtruth images having the same same as the labels. To better understand, if there is a label path/to/labels/000001.txt there should be a corresponding image path/to/images/000001.jpg. 
* ```<config file path>```: path to a yaml file with the parameters needed for the mAP computation, similar to demo/config.yaml

Example:

```
cd build
./map_demo dla34_cnet.rt c ../demo/COCO_val2017/all_labels.txt ../demo/config.yaml
```

## Supported networks

| Test Name         | Network                                       | Dataset                                                       | N Classes | Input size    | Weights                                                                   |
| :---------------- | :-------------------------------------------- | :-----------------------------------------------------------: | :-------: | :-----------: | :------------------------------------------------------------------------ |
| yolo              | YOLO v2<sup>1</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 608x608       | weights                                                                   |
| yolo_224          | YOLO v2<sup>1</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| yolo_berkeley     | YOLO v2<sup>1</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)   | 10        | 416x736       | weights                                                                   |
| yolo_relu         | YOLO v2 (with ReLU, not Leaky)<sup>1</sup>    | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | weights                                                                   |
| yolo_tiny         | YOLO v2 tiny<sup>1</sup>                      | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | weights                                                                   |
| yolo_voc          | YOLO v2<sup>1</sup>                           | [VOC      ](http://host.robots.ox.ac.uk/pascal/VOC/)          | 21        | 416x416       | weights                                                                   |
| yolo3             | YOLO v3<sup>2</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/jPXmHyptpLoNdNR/download)     |
| yolo3_berkeley    | YOLO v3<sup>2</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)   | 10        | 320x544       | weights                                                                   |
| yolo3_coco4       | YOLO v3<sup>2</sup>                           | [COCO 2014](http://cocodataset.org/)                          | 4         | 416x416       | weights                                                                   |
| yolo3_flir        | YOLO v3<sup>2</sup>                           | [FREE FLIR](https://www.flir.com/oem/adas/adas-dataset-form/) | 3         | 320x544       | weights                                                                   |
| yolo3_tiny        | YOLO v3 tiny<sup>2</sup>                      | [COCO 2014](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/LMcSHtWaLeps8yN/download)     |
| yolo3_tiny512     | YOLO v3 tiny<sup>2</sup>                      | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/pjooA5DMrrEbrmA/download)     |
| dla34             | Deep Leayer Aggreagtion (DLA) 34<sup>3</sup>  | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| dla34_cnet        | Centernet (DLA34 backend)<sup>4</sup>         | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/8AjXdgCeRzCa5AF/download)     |
| mobilenetv2ssd    | Mobilnet v2 SSD Lite<sup>5</sup>              | [VOC      ](http://host.robots.ox.ac.uk/pascal/VOC/)          | 21        | 300x300       | [weights](https://cloud.hipert.unimore.it/s/x4ZfxBKN23zAJQp/download)     |
| resnet101         | Resnet 101<sup>6</sup>                        | [COCO 2014](http://cocodataset.org/)                          | 80        | 224x224       | weights                                                                   |
| resnet101_cnet    | Centernet (Resnet101 backend)<sup>4</sup>     | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/B6mj33k7beECXsY/download)     |



## References

1. Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
3. Yu, Fisher, et al. "Deep layer aggregation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
4. Zhou, Xingyi, Dequan Wang, and Philipp Krähenbühl. "Objects as points." arXiv preprint arXiv:1904.07850 (2019).
5. Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
6. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.