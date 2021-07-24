# tkDNN export weights

## Index 

 - [How to export weights](#how-to-export-weights)
    - [1)Export weights from darknet](#1export-weights-from-darknet)
    - [2)Export weights for DLA34 and ResNet101](#2export-weights-for-dla34-and-resnet101)
    - [3)Export weights for CenterNet](#3export-weights-for-centernet)
    - [4)Export weights for MobileNetSSD](#4export-weights-for-mobilenetssd)
    - [5)Export weights for CenterTrack](#5export-weights-for-centertrack)
    - [6)Export weights for ShelfNet](#6export-weights-for-shelfnet)
 - [Darknet Parser](#darknet-parser)

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
N.B. Use compilation with CPU (leave GPU=0 in Makefile) if you also want debug. 

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
### 5)Export weights for CenterTrack
To get the weights needed to run CenterTrack tests use [this](https://github.com/sapienzadavide/CenterTrack.git) fork of the original CenterTrack. 
```
git clone https://github.com/sapienzadavide/CenterTrack.git
```
* follow the instruction in the README.md and INSTALL.md

```
python demo.py tracking,ddd --load_model ../models/nuScenes_3Dtracking.pth --dataset nuscenes --pre_hm --track_thresh 0.1 --demo /path/to/image/or/folder/or/video/or/webcam --test_focal_length 633 --exp_wo --exp_wo_dim 512 --input_h 512 --input_w 512
```

### 6)Export weights for ShelfNet
To get the weights needed to run Shelfnet tests use [this](https://git.hipert.unimore.it/mverucchi/shelfnet) fork of a Pytorch implementation of Shelfnet network. 

```
git clone https://git.hipert.unimore.it/mverucchi/shelfnet
cd shelfnet 
cd ShelfNet18_realtime
conda env create --file shelfnet_env.yml
conda activate shelfnet
mkdir layer debug
python export.py
```

## Darknet Parser
tkDNN implement and easy parser for darknet cfg files, a network can be converted with *tk::dnn::darknetParser*:
```
// example of parsing yolo4
tk::dnn::Network *net = tk::dnn::darknetParser("yolov4.cfg", "yolov4/layers", "coco.names");
net->print();
```
All models from darknet are now parsed directly from cfg, you still need to export the weights with the described tools in the previous section.
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
  logistic
</details>