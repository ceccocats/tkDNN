# tkDNN
tkDNN is a Deep Neural Network library built with cuDNN and tensorRT primitives, specifically thought to work on NVIDIA Jetson Boards. It has been tested on TK1(branch cudnn2), TX1, TX2, AGX Xavier, Nano and several discrete GPUs.
The main goal of this project is to exploit NVIDIA boards as much as possible to obtain the best inference performance. It does not allow training. 


If you use tkDNN in your research, please cite the [following paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9212130&casa_token=sQTJXi7tJNoAAAAA:BguH9xCIY48MxbtDS3LXzIXzO-9sWArm7Hd7y7BwaLmqRuM_Gx8bOYizFPNMNtpo5K0kB-P-). For use in commercial solutions, write at gattifrancesco@hotmail.it and micaela.verucchi@unimore.it or refer to https://hipert.unimore.it/ .

```
@inproceedings{verucchi2020systematic,
  title={A Systematic Assessment of Embedded Neural Networks for Object Detection},
  author={Verucchi, Micaela and Brilli, Gianluca and Sapienza, Davide and Verasani, Mattia and Arena, Marco and Gatti, Francesco and Capotondi, Alessandro and Cavicchioli, Roberto and Bertogna, Marko and Solieri, Marco},
  booktitle={2020 25th IEEE International Conference on Emerging Technologies and Factory Automation (ETFA)},
  volume={1},
  pages={937--944},
  year={2020},
  organization={IEEE}
}
```

### What's new (20 July 2021)
- [x] Support to sematic segmentation [README](docs/README_seg.md)
- [x] Support 2D/3D Object Detection and Tracking [README](docs/README_2d3dtracking.md)
- [ ] Support to TensorRT8 (WIP)

## FPS Results
Inference FPS of yolov4 with tkDNN, average of 1200 images with the same dimension as the input size, on 
  * RTX 2080Ti (CUDA 10.2, TensorRT 7.0.0, Cudnn 7.6.5);
  * Xavier AGX, Jetpack 4.3 (CUDA 10.0, CUDNN 7.6.3, tensorrt 6.0.1 );
  * Xavier NX, Jetpack 4.4  (CUDA 10.2, CUDNN 8.0.0, tensorrt 7.1.0 ). 
  * Tx2, Jetpack 4.2 (CUDA 10.0, CUDNN 7.3.1, tensorrt 5.0.6 );
  * Jetson Nano, Jetpack 4.4  (CUDA 10.2, CUDNN 8.0.0, tensorrt 7.1.0 ). 

| Platform   | Network    | FP32, B=1 | FP32, B=4	| FP16, B=1 |	FP16, B=4 |	INT8, B=1 |	INT8, B=4 | 
| :------:   | :-----:    | :-----:   | :-----:   | :-----:   |	:-----:   |	:-----:   |	:-----:   | 
| RTX 2080Ti | yolo4 320  | 118.59	  | 237.31	  | 207.81	  | 443.32	  | 262.37	  | 530.93    | 
| RTX 2080Ti | yolo4 416  | 104.81	  | 162.86	  | 169.06	  | 293.78	  | 206.93	  | 353.26    | 
| RTX 2080Ti | yolo4 512  | 92.98	    | 132.43	  | 140.36	  | 215.17	  | 165.35	  | 254.96    | 
| RTX 2080Ti | yolo4 608  | 63.77	    | 81.53	    | 111.39	  | 152.89	  | 127.79	  | 184.72    | 
| AGX Xavier | yolo4 320  |	26.78	    | 32.05	    | 57.14	    | 79.05	    | 73.15	    | 97.56     |
| AGX Xavier | yolo4 416  |	19.96	    | 21.52	    | 41.01	    | 49.00	    | 50.81	    | 60.61     |
| AGX Xavier | yolo4 512  |	16.58	    | 16.98	    | 31.12	    | 33.84	    | 37.82	    | 41.28     |
| AGX Xavier | yolo4 608  |	9.45 	    | 10.13	    | 21.92	    | 23.36	    | 27.05	    | 28.93     |
| Xavier NX  | yolo4 320  |	14.56	    | 16.25	    | 30.14	    | 41.15	    | 42.13	    | 53.42     |
| Xavier NX  | yolo4 416  |	10.02	    | 10.60	    | 22.43	    | 25.59	    | 29.08	    | 32.94     |
| Xavier NX  | yolo4 512  |	8.10	    | 8.32	    | 15.78	    | 17.13	    | 20.51	    | 22.46     |
| Xavier NX  | yolo4 608  |	5.26	    | 5.18	    | 11.54	    | 12.06	    | 15.09	    | 15.82     |
| Tx2        | yolo4 320	| 11.18	    | 12.07	    | 15.32	    | 16.31     | -         | -         |
| Tx2        | yolo4 416	| 7.30	    | 7.58	    | 9.45	    | 9.90      | -         | -         |
| Tx2        | yolo4 512	| 5.96	    | 5.95	    | 7.22	    | 7.23      | -         | -         |
| Tx2        | yolo4 608	| 3.63	    | 3.65	    | 4.67	    | 4.70      | -         | -         |
| Nano       | yolo4 320	| 4.23	    | 4.55	    | 6.14	    | 6.53      | -         | -         |
| Nano       | yolo4 416	| 2.88	    | 3.00	    | 3.90	    | 4.04      | -         | -         |
| Nano       | yolo4 512	| 2.32	    | 2.34	    | 3.02	    | 3.04      | -         | -         |
| Nano       | yolo4 608	| 1.40	    | 1.41	    | 1.92	    | 1.93      | -         | -         |

## MAP Results
Results for COCO val 2017 (5k images), on RTX 2080Ti, with conf threshold=0.001

|                      | CodaLab       | CodaLab   | CodaLab       | CodaLab     | tkDNN map     | tkDNN map |
| -------------------- | :-----------: | :-------: | :-----------: | :---------: | :-----------: | :-------: |
|                      | **tkDNN**     | **tkDNN** | **darknet**   | **darknet** | **tkDNN**     | **tkDNN** |
|                      | MAP(0.5:0.95) | AP50      | MAP(0.5:0.95) | AP50        | MAP(0.5:0.95) | AP50      |
| Yolov3 (416x416)     | 0.381         | 0.675     | 0.380         | 0.675       | 0.372         | 0.663     |
| yolov4 (416x416)     | 0.468         | 0.705     | 0.471         | 0.710       | 0.459         | 0.695     |
| yolov3tiny (416x416) | 0.096         | 0.202     | 0.096         | 0.201       | 0.093         | 0.198     |
| yolov4tiny (416x416) | 0.202         | 0.400     | 0.201         | 0.400       | 0.197         | 0.395     |
| Cnet-dla34 (512x512) | 0.366         | 0.543     | \-            | \-          | 0.361         | 0.535     |
| mv2SSD (512x512)     | 0.226         | 0.381     | \-            | \-          | 0.223         | 0.378     |

## Index
- [tkDNN](#tkdnn)
  - [Index](#index)
  - [Dependencies](#dependencies)
  - [How to compile this repo](#how-to-compile-this-repo)
  - [Workflow](#workflow)
  - [Exporting weights](#exporting-weights)
  - [Run the demos](#run-the-demos)
  - [tkDNN on Windows 10 (experimental)](#tkdnn-on-windows-10-experimental)
  - [Existing tests and supported networks](#existing-tests-and-supported-networks)
  - [References](#references)
  

## Dependencies
This branch works on every NVIDIA GPU that supports the following (latest tested) dependencies:
* CUDA 11.0 (or >= 10) [the segmentation only works with CUDA 10 for now]
* cuDNN 8.0.4 (or >= 7.3)
* TensorRT 7.2.0 (or >=5)
* OpenCV 4.5.2 (or >=4)
* cmake 3.21 (or >= 3.15)
* yaml-cpp 0.5.2
* eigen3 3.3.4
* curl 7.58

```
sudo apt install libyaml-cpp-dev curl libeigen3-dev

```

#### About OpenCV
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

## Exporting weights

For specific details on how to export weights see [HERE](./docs/exporting_weights.md).

## Run the demos 

For specific details on how to run:
- 2D object detection demos, details on FP16, INT8 and batching see [HERE](./docs/demo.md).
- segmentation demos see [HERE](./docs/README_seg.md).
- 2D/3D object detection and tracking demos see [HERE](./docs/README_2d3dtracking.md).
- mAP demo to evaluate 2D object detectors see [HERE](./docs/mAP_demo.md).

![demo](https://user-images.githubusercontent.com/11562617/72547657-540e7800-388d-11ea-83c6-49dfea2a0607.gif)

## tkDNN on Windows 10 (experimental)

For specific details on how to run tkDNN on Windows 10 see [HERE](./docs/windows.md).

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
| yolo4_320         | Yolov4 <sup>8</sup>                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 320x320       | [weights](https://cloud.hipert.unimore.it/s/d97CFzYqCPCp5Hg/download)     |
| yolo4_512         | Yolov4 <sup>8</sup>                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/d97CFzYqCPCp5Hg/download)     |
| yolo4_608         | Yolov4 <sup>8</sup>                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 608x608       | [weights](https://cloud.hipert.unimore.it/s/d97CFzYqCPCp5Hg/download)     |
| yolo4_berkeley             | Yolov4 <sup>8</sup>                           | [BDD100K  ](https://bair.berkeley.edu/blog/2018/05/30/bdd/)                          | 10        | 540x320       | [weights](https://cloud.hipert.unimore.it/s/nkWFa5fgb4NTdnB/download)     |
| yolo4tiny             | Yolov4 tiny <sup>9</sup>                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 416x416       | [weights](https://cloud.hipert.unimore.it/s/iRnc4pSqmx78gJs/download)     |
| yolo4x             | Yolov4x-mish  <sup>9</sup>                          | [COCO 2017](http://cocodataset.org/)                          | 
| yolo4tiny_512           | Yolov4 tiny <sup>9</sup>                           | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/iRnc4pSqmx78gJs/download)     |
80        | 640x640       | [weights](https://cloud.hipert.unimore.it/s/5MFjtNtgbDGdJEo/download)     |
| yolo4x-cps            | Scaled Yolov4 <sup>10</sup>                          | [COCO 2017](http://cocodataset.org/)                          | 80        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/AfzHE4BfTeEm2gH/download)     |
| shelfnet              | ShelfNet18_realtime<sup>11</sup>                      | [Cityscapes](https://www.cityscapes-dataset.com/)                          | 19        | 1024x1024       | [weights](https://cloud.hipert.unimore.it/s/mEDZMRJaGCFWSJF/download)                                                                   |
| shelfnet_berkeley              | ShelfNet18_realtime<sup>11</sup>                           | [DeepDrive](https://bdd-data.berkeley.edu/)                          | 20        | 1024x1024       | [weights](https://cloud.hipert.unimore.it/s/m92e7QdD9gYMF7f/download)                                                                   |
| dla34_cnet3d        | Centernet3D (DLA34 backend)<sup>4</sup>         | [KITTI 2017](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)                          | 1        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/2MDyWGzQsTKMjmR/download)     |
| dla34_ctrack        | CenterTrack (DLA34 backend)<sup>12</sup>         | [NuScenes 3D](https://www.nuscenes.org/)                          | 7        | 512x512       | [weights](https://cloud.hipert.unimore.it/s/rjNfgGL9FtAXLHp/download)     |


## References

1. Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
2. Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
3. Yu, Fisher, et al. "Deep layer aggregation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
4. Zhou, Xingyi, Dequan Wang, and Philipp Krähenbühl. "Objects as points." arXiv preprint arXiv:1904.07850 (2019).
5. Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
6. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
7. Wang, Chien-Yao, et al. "CSPNet: A New Backbone that can Enhance Learning Capability of CNN." arXiv preprint arXiv:1911.11929 (2019).
8. Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "YOLOv4: Optimal Speed and Accuracy of Object Detection." arXiv preprint arXiv:2004.10934 (2020).
9. Bochkovskiy, Alexey, "Yolo v4, v3 and v2 for Windows and Linux" (https://github.com/AlexeyAB/darknet)
10. Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "Scaled-YOLOv4: Scaling Cross Stage Partial Network." arXiv preprint arXiv:2011.08036 (2020).
11. Zhuang, Juntang, et al. "ShelfNet for fast semantic segmentation." Proceedings of the IEEE International Conference on Computer Vision Workshops. 2019.
12. Zhou, Xingyi, Vladlen Koltun, and Philipp Krähenbühl. "Tracking objects as points." European Conference on Computer Vision. Springer, Cham, 2020.
