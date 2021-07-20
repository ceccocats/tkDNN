# Use the prebuilt image
```
# build image
docker build -t tkdnn:build -f Dockerfile . 
```

# Build Base Docker image
```
# make nvidia docker working
# follow this guide: https://github.com/NVIDIA/nvidia-docker

# dowload tensorrt
# from: https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.0/7.0.0.11/local_repo/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb

# build image
docker build -t ceccocats/tkdnn:latest -f Dockerfile.base .

# run image
docker run -ti --gpus all --rm ceccocats/tkdnn:latest bash
```

