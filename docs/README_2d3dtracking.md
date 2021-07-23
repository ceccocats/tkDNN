# 2D/3D Object Detection and Tracking

Currently tkDNN supports only CenterTrack as 3DOD & 2D/3D Tracker network.

## 3D Object Detection

To run the 3D object detection demo follow these steps (example with CenterNet based on DLA34):
```
rm dla34_cnet3d_fp32.rt        # be sure to delete(or move) old tensorRT files
./test_dla34_cnet3d            # run the yolo test (is slow)
./demo3D dla34_cnet3d_fp32.rt ../demo/yolo_test.mp4 NULL c
```
The demo3D program takes the same parameters of the demo program:
```
./demo3D <network-rt-file> <path-to-video> <calibration-file> <kind-of-network> <number-of-classes> <n-batches> <show-flag> <conf-thresh>
```
where

* ```<calibration-file>``` is the camera calibration file (opencv format). It is important that the file contains entry "camera_matrix" with sub-entry "rows", "cols", "data". If you do not want to pass the calibration file, pass "NULL" instead.

![demo](https://user-images.githubusercontent.com/11939259/126784875-c4285497-d369-424f-abda-58274cd747ac.gif)

## Object Detection and Tracking

To run the 3D object detection & tracking demo follow these steps (example with CenterTrack based on DLA34):
```
rm dla34_ctrack_fp32.rt  # be sure to delete(or move) old tensorRT files
./test_dla34_ctrack      # run the yolo test (is slow)
./demoTracker dla34_ctrack_fp32.rt ../demo/yolo_test.mp4 NULL c
```

The demoTracker program takes the same parameters of the demo program:
```
./demoTracker <network-rt-file> <path-to-video> <calibration-file> <kind-of-network> <number-of-classes> <n-batches> <show-flag> <conf-thresh> <2D/3D-flag>
```

where

* ```<calibration-file>``` is the camera calibration file (opencv format). It is important that the file contains entry "camera_matrix" with sub-entry "rows", "cols", "data". If you do not want to pass the calibration file, pass "NULL" instead.
*  ```<2D/3D-flag>``` if set to 0 the demo will be in the 2D mode, while if set to 1 the demo will be in the 3D mode (Default is 1 - 3D mode).

![demo](https://user-images.githubusercontent.com/11939259/126784878-513fa9e8-864a-4c24-b4bd-199737184708.gif)

## FPS Results

Inference FPS of shelfnet with tkDNN, average of 1200 images on:
  * RTX 2080Ti (CUDA 10.2, TensorRT 7.0.0, Cudnn 7.6.5);
  * Xavier AGX, Jetpack 4.3 (CUDA 10.0, CUDNN 7.6.3, tensorrt 6.0.1 );

### 3D OD and Tracking

| Platform   | Test                        | Phase   | FP32, ms  | FP32, FPS | FP16, ms  |	FP16, FPS | INT8, ms  | INT8, FPS | 
| :------:   | :-----:                     | :-----: | :-----:   | :-----:   | :-----:   |	:-----:   | :-----:   | :-----:   | 
| RTX 2080Ti | CenterTrack3D 512x512 (B=1) | pre     | 4.43883   |  225.285  |  4.42951  |  225.759   |  4.44278  |  225.084  |
| RTX 2080Ti | CenterTrack3D 512x512 (B=1) | inf     | 9.03454   |  110.686  |  6.02013  |  166.109   |  5.31611  |  188.108  |
| RTX 2080Ti | CenterTrack3D 512x512 (B=1) | post    | 0.96631   |  1034.87  |  0.96824  |  1032.80   |  0.95066  |  1051.90  |
| RTX 2080Ti | CenterTrack3D 512x512 (B=1) | tot     | 14.4397   |  69.2535  |  11.4179  |  87.5818   |  10.7095  |  93.3750  |
| RTX 2080Ti | CenterTrack3D 512x512 (B=4) | pre     | 4.60075   |  217.356  |  4.28658  |  233.286   |  4.29473  |  232.844  | 
| RTX 2080Ti | CenterTrack3D 512x512 (B=4) | inf     | 8.48365   |  117.874  |  5.25150  |  190.422   |  4.58463  |  218.120  |  
| RTX 2080Ti | CenterTrack3D 512x512 (B=4) | post    | 0.99484   |  1005.19  |  0.91776  |  1089.61   |  0.89853  |  1112.93  |  
| RTX 2080Ti | CenterTrack3D 512x512 (B=4) | tot     | 14.0792   |  71.0266  |  10.4558  |  95.6405   |  9.77788  |  102.272  |  
| AGX Xavier | CenterTrack3D 512x512 (B=1) | pre     | 34.9915   |  28.5784  |  33.5976  |  29.7440   |  34.4425  |  29.0339  |
| AGX Xavier | CenterTrack3D 512x512 (B=1) | inf     | 76.3579   |  13.0962  |  52.4759  |  19.0564   |  51.4610  |  19.4322  |
| AGX Xavier | CenterTrack3D 512x512 (B=1) | post    | 3.38576   |  295.355  |  3.26010  |  306.739   |  3.19770  |  312.725  |
| AGX Xavier | CenterTrack3D 512x512 (B=1) | tot     | 114.735   |  8.71574  |  89.3336  |  11.1940   |  89.1012  |  11.2232  |
| AGX Xavier | CenterTrack3D 512x512 (B=4) | pre     | 32.8933   |  30.4014  |  32.7950  |  30.4925   |  32.9603  |  30.3396  | 
| AGX Xavier | CenterTrack3D 512x512 (B=4) | inf     | 74.2840   |  13.4618  |  50.3858  |  19.8469   |  49.2030  |  20.3240  |  
| AGX Xavier | CenterTrack3D 512x512 (B=4) | post    | 3.14888   |  317.574  |  3.13615  |  318.862   |  3.02550  |  330.524  |  
| AGX Xavier | CenterTrack3D 512x512 (B=4) | tot     | 110.326   |  9.06404  |  86.3169  |  11.5852   |  85.1888  |  11.7386  |  


### 2D OD and Tracking

| Platform   | Test                        | Phase   | FP32, ms  | FP32, FPS | FP16, ms  |	FP16, FPS | INT8, ms  | INT8, FPS | 
| :------:   | :-----:                     | :-----: | :-----:   | :-----:   | :-----:   |	:-----:   | :-----:   | :-----:   | 
| RTX 2080Ti | CenterTrack2D 512x512 (B=1) | pre     | 4.44386   |  225.030  |  4.43828  |  225.313   |  4.47747  |  223.340  |
| RTX 2080Ti | CenterTrack2D 512x512 (B=1) | inf     | 9.08365   |  110.088  |  6.04842  |  165.332   |  5.34787  |  186.990  |
| RTX 2080Ti | CenterTrack2D 512x512 (B=1) | post    | 0.98593   |  1014.27  |  0.97745  |  1023.07   |  0.96595  |  1035.25  |
| RTX 2080Ti | CenterTrack2D 512x512 (B=1) | tot     | 14.5134   |  68.9018  |  11.4642  |  87.2281   |  10.7913  |  92.6672  |
| RTX 2080Ti | CenterTrack2D 512x512 (B=4) | pre     | 4.41188   |  226.661  |  4.50800  |  221.828   |  4.29238  |  232.971  | 
| RTX 2080Ti | CenterTrack2D 512x512 (B=4) | inf     | 8.29015   |  120.625  |  5.38630  |  185.656   |  4.58500  |  218.103  |  
| RTX 2080Ti | CenterTrack2D 512x512 (B=4) | post    | 0.96847   |  1032.55  |  0.97997  |  1020.44   |  0.91791  |  1089.43  |  
| RTX 2080Ti | CenterTrack2D 512x512 (B=4) | tot     | 13.6705   |  73.1502  |  10.8743  |  91.9602   |  9.79528  |  102.090  |  
| AGX Xavier | CenterTrack2D 512x512 (B=1) | pre     | 33.4745   |  29.8735  |  33.4847  |  29.8643   |  33.5022  |  29.8488  |
| AGX Xavier | CenterTrack2D 512x512 (B=1) | inf     | 76.2077   |  13.1220  |  52.5111  |  19.0436   |  51.6057  |  19.3777  |
| AGX Xavier | CenterTrack2D 512x512 (B=1) | post    | 3.26055   |  306.697  |  3.26806  |  305.992   |  3.21988  |  310.571  |
| AGX Xavier | CenterTrack2D 512x512 (B=1) | tot     | 111.943   |  8.93312  |  89.2639  |  11.2027   |  88.3278  |  11.3215  |
| AGX Xavier | CenterTrack2D 512x512 (B=4) | pre     | 32.8323   |  30.4579  |  32.8595  |  30.4326   |  32.8195  |  30.4697  | 
| AGX Xavier | CenterTrack2D 512x512 (B=4) | inf     | 74.3075   |  13.4576  |  50.3555  |  19.8588   |  49.1805  |  20.3333  |  
| AGX Xavier | CenterTrack2D 512x512 (B=4) | post    | 3.12360   |  320.143  |  3.13570  |  318.908   |  3.04943  |  327.931  |  
| AGX Xavier | CenterTrack2D 512x512 (B=4) | tot     | 110.263   |  9.06920  |  86.3507  |  11.5807   |  85.0494  |  11.7579  |  

