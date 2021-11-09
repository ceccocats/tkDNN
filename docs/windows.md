# tkDNN on Windows 

## Index

 - [Dependencies-Windows](#dependencies-windows)
 - [Compiling tkDNN on Windows](#compiling-tkdnn-on-windows)
 - [Run the demo on Windows](#run-the-demo-on-windows)
    - [FP16 inference windows](#fp16-inference-windows)
    - [INT8 inference windows](#int8-inference-windows)
 - [Run tkDNN on WSL2 with cuda](#tkdnn-on-cuda-wsl)
 - [Known issues with tkDNN on Windows](#known-issues-with-tkdnn-on-windows)

### Dependencies-Windows 
This branch should work on every NVIDIA GPU supported in windows with the following dependencies:

* WINDOWS 10 1803 or HIGHER 
* CUDA 11.2 
* CUDNN 8.1.1
* TENSORRT 7.2.3 
* OPENCV 4.2  
* MSVC 16.9+
* YAML-CPP 
* EIGEN3
* 7ZIP (ADD TO PATH)
* NINJA 1.10


All the above mentioned dependencies except 7ZIP can be installed using Microsoft's [VCPKG](https://github.com/microsoft/vcpkg.git) .
After bootstrapping VCPKG the dependencies can be built and installed using the following command :

```
opencv4(normal) - vcpkg.exe install opencv4[tbb,jpeg,tiff,opengl,openmp,png,ffmpeg,eigen]:x64-windows yaml-cpp:x64-windows eigen3:x64-windows --x-install-root=C:\opt --x-buildtrees-root=C:\temp_vcpkg_build

opencv4(cuda) - vcpkg.exe install opencv4[cuda,nonfree,contrib,eigen,tbb,jpeg,tiff,opengl,openmp,png,ffmpeg]:x64-windows yaml-cpp:x64-windows eigen3:x64-windows --x-install-root=C:\opt --x-buildtrees-root=C:\temp_vcpkg_build
```
To build opencv4 with cuda and cudnn version corresponding to your cuda version,vcpkg's cudnn portfile needs to be modified by adding ```$ENV{CUDA_PATH}```  at lines 16 and 17 in the portfile.cmake 

After VCPKG finishes building and installing all the packages delete C:\temp_vcpkg_build and add C:\opt\x64-windows\bin and C:\opt\x64-windows\debug\bin to path 

### Compiling tkDNN on Windows 

tkDNN is built with cmake(3.15+) on windows along with ninja.Msbuild and NMake Makefiles are drastically slower when compiling the library compared to windows
```
git clone https://github.com/ceccocats/tkDNN.git
cd tkdnn-windows
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -G"Ninja" ..
ninja -j4
```

### Run the demo on Windows 

This example uses yolo4_tiny.\
To run the object detection file create .rt file bu running:
```
.\test_yolo4tiny.exe
```

Once the rt file has been successfully create,run the demo using the following command:
```
.\demo.exe yolo4_fp32.rt ..\demo\yolo_test.mp4 y 80 ..\tests\darknet\cfg\yolo4.cfg ..\tests\darknet\names\cococ.names 
```
 For general info on more demo paramters,check Run the demo section on top 
 To run the test_all_tests.sh on windows,use git bash or msys2 

### FP16 inference windows 

This is an untested feature on windows.To run the object detection demo with FP16 interference follow the below steps(example with yolo4tiny):
```
set TKDNN_MODE=FP16
del /f yolo4tiny_fp16.rt
.\test_yolo4tiny.exe
.\demo.exe yolo4tiny_fp16.rt ..\demo\yolo_test.mp4
```

### INT8 inference windows 
To run object detection demo with INT8 (example with yolo4tiny):
```
set TKDNN_MODE=INT8
set TKDNN_CALIB_LABEL_PATH=..\demo\COCO_val2017\all_labels.txt
set TKDNN_CALIB_IMG_PATH=..\demo\COCO_val2017\all_images.txt
del /f  yolo4tiny_int8.rt        # be sure to delete(or move) old tensorRT files
.\test_yolo4tiny.exe           # run the yolo test (is slow)
.\demo.exe yolo4tiny_int8.rt ..\demo\yolo_test.mp4 y

```

### Run tkDNN on WSL2 with cuda
tkDNN works on wsl2 with cuda,although not all networks (centernet,mobilenet) work properly.
If you encounter issues with running the network as a result of driver not found or cuda launch error,running the following command should solve the issue
```cp /usr/lib/wsl/lib/lib* /usr/lib/x86_64-linux-gnu/ ```



### Known issues with tkDNN on Windows

In theory all models (centernet,mobilenet,darknet,centertrack,cnet3d and shelfnet) should work on Windows.

On pascal cards(sm 6x) ,nvidia cuda wsl driver 510.06 don't work well with tkDNN both on windows and cuda wsl , Nvidia drivers >465+ and < 500 are completely supported .


