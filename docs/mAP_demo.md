# Run the mAP demo

To compute mAP, precision, recall and f1score to evaluate 2D object detectors, run the map_demo.

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

This demo also creates a json file named ```net_name_COCO_res.json``` containing all the detections computed. The detections are in COCO format, the correct format to submit the results to [CodaLab COCO detection challenge](https://competitions.codalab.org/competitions/20794#participate).