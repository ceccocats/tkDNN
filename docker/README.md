# Use the prebuilt image
```
# build image
docker build -t tkdnn:build -f Dockerfile . 
```

# Build Base Docker image
```
# make nvidia docker working
# follow this guide: https://github.com/NVIDIA/nvidia-docker

# build image
docker build -t ceccocats/tkdnn:latest -f Dockerfile.base .

# run image
./docker_launch.sh
```

