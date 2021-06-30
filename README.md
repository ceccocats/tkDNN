
# Steps to build docker image
## Docker
Docker version 19.03 will be required.

## For running with GPU
### NVIDIA Drivers
This drivers should be installed in the host system.
### For Ubuntu:
```
$ sudo apt-get install linux-headers-$(uname -r) gcc g++ make
$ wget http://in.download.nvidia.com/tesla/418.67/NVIDIA-Linux-x86_64-418.67.run
$ chmod 777 NVIDIA-Linux-x86_64-418.67.run
$ bash NVIDIA-Linux-x86_64-418.67.run
```

### For CentOS
```
$ sudo yum -y install kernel-devel-$(uname -r) kernel-header-$(uname -r) gcc make
$ wget http://in.download.nvidia.com/tesla/418.67/NVIDIA-Linux-x86_64-418.67.run
$ chmod 777 NVIDIA-Linux-x86_64-418.67.run
$ bash NVIDIA-Linux-x86_64-418.67.run
```

### NVIDIA Container Toolkit
This toolkit should be installed in the host system.
### For Ubuntu
```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

### For CentOS
```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

$ sudo yum install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

### Building docker image

* Go to ``TKDNN/`` then run following command.
      ```$ docker build -t baggageai:server -f docker/Dockerfile .```
  #### Note:
1. have to copy weights into a TKDNN/config/ (tkdnn converted weights) current support api (fp32X4 and fp16X1) 
2. setup number of classes accrding to weights in TKDNN/config/config.yml
3. give path of this weights into handler.cpp line number 117-121.

### Run docker image
```
$ docker run --gpus all -p 8080:8080 -d <image_id>
```
Now server will be started in the container. You can check server is running or not using ``docker ps``




### Run docker image
```
$ docker run -p 8080:8080 -d <image_id>
```


# Using docker-compose file
##  Docker Compose
Install docker-compose version 1.24.1
[https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)

## Create Volume
Create a volume named ``BAI_logs`` using following command:
```
$ docker volume create BAI_logs
```
Change permission of the directory of docker volume, so that logs can be written to that directory.
```
$ cd /var/lib/docker/volumes/BAI_logs
$ chmod 757 _data/
```
```
## For running with GPU
Install nvidia-container-runtime:
```
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
$ sudo apt-get update
$ sudo apt-get install nvidia-container-runtime
```

Add nvidia runtime in ``/etc/docker/daemon.json ``
```
{
  "runtimes": {
                "nvidia": {
                        "path": "/usr/bin/nvidia-container-runtime",
                        "runtimeArgs": []
                }
        },
  "default-runtime": "nvidia"

}
```
After editing changes restart the docker.
``systemctl restart docker ``

Go to ``BaggageAI-Darknet-API/baggageai/dist/server/with-gpu`` and run:
```
$ docker-compose up -d
```

## Running containers in stack
First of all initialize a swarm.
```
$ docker swarm init
```
You can add a worker node using ``docker swarm join`` command displayed on terminal. 


Check the statsus using
```
$ docker service ls
$ docker stack ls
```

### With GPU
Go to ``TKDNN/docker/`` and run:
```
$ docker stack deploy -c docker-compose.yml <service-name>
```

# Calling the API 
```
curl -X POST http://localhost:8080?name=<file_name> --data-binary "@<absolute_path_of_image>"
```
Example,
```
curl -X POST http://localhost:8080?name=S0240628297_20180812164749_L-4_3.jpg \
--data-binary "@/home/ubuntu/BaggageAI/S0240628297_20180812164749_L-4_3.jpg"
```

