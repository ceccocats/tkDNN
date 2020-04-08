#!/bin/bash

cd build

RED='\033[1;31m'
GREEN='\033[1;32m'
ORANGE='\033[1;33m'
NC='\033[0m' # No Color

function print_output {
    if [ $1 -eq 0 ]
    then
        echo -e "$2 ${GREEN}OK${NC}"
    else
        echo -e "$2 ${RED}NOT OKAY${NC}"
    fi 

} 

out_file=results.log
rm $out_file

modes=( 1 ) # only FP32
# modes=( 1 2 ) # FP32 and FP16
# modes=( 1 2 3 ) # FP32, FP16 and INT8

for i in "${modes[@]}"
do
    rm *rt
    if [ $i -eq 1 ]
    then
        export TKDNN_MODE=FP32
        echo -e "${ORANGE}Test FP32${NC}"
    fi
    if [ $i -eq 2 ]
    then
        export TKDNN_MODE=FP16
        echo -e "${ORANGE}Test FP16${NC}"
    fi
    if [ $i -eq 3 ]
    then
        export TKDNN_MODE=INT8
        export TKDNN_CALIB_LABEL_PATH=../demo/COCO_val2017/all_labels.txt
        export TKDNN_CALIB_IMG_PATH=../demo/COCO_val2017/all_images.txt
        echo -e "${ORANGE}Test INT8${NC}"
	fi

    ./test_imuodom &>> $out_file
    res_imuodom=$?
    print_output $res_imuodom test_imuodom

    ./test_resnet101_cnet &>> $out_file
    res_resnet101_cnet=$?
    print_output $res_resnet101_cnet test_resnet101_cnet

    ./test_yolo3 &>> $out_file
    res_yolo3=$?
    print_output $res_yolo3 test_yolo3

    ./test_yolo3_flir &>> $out_file
    res_yolo3_flir=$?
    print_output $res_yolo3_flir test_yolo3_flir

    ./test_yolo3_512 &>> $out_file
    res_yolo3_512=$?
    print_output $res_yolo3_512 test_yolo3_512

    ./test_yolo3_tiny &>> $out_file
    res_yolo3_tiny=$?
    print_output $res_yolo3_tiny test_yolo3_tiny

    ./test_csresnext50-panet-spp &>> $out_file
    res_csresnext50panetspp=$?
    print_output $res_csresnext50panetspp "test_csresnext50-panet-spp"

    ./test_mobilenetv2ssd &>> $out_file
    res_mobilenetv2ssd=$?
    print_output $res_mobilenetv2ssd test_mobilenetv2ssd

    ./test_yolo3_tiny512 &>> $out_file
    res_yolo3_tiny512=$?
    print_output $res_yolo3_tiny512 test_yolo3_tiny512

    ./test_yolo_tiny &>> $out_file
    res_yolo_tiny=$?
    print_output $res_yolo_tiny test_yolo_tiny

    ./test_mobilenetv2ssd512 &>> $out_file
    res_mobilenetv2ssd512=$?
    print_output $res_mobilenetv2ssd512 test_mobilenetv2ssd512

    ./test_mnist &>> $out_file
    res_mnist=$?
    print_output $res_mnist test_mnist

    ./test_yolo &>> $out_file
    res_yolo=$?
    print_output $res_yolo test_yolo

    ./test_yolo3_berkeley &>> $out_file
    res_yolo3_berkeley=$?
    print_output $res_yolo3_berkeley test_yolo3_berkeley

    ./test_yolo_voc &>> $out_file
    res_yolo_voc=$?
    print_output $res_yolo_voc test_yolo_voc

    ./test_dla34_cnet &>> $out_file
    res_dla34_cnet=$?
    print_output $res_dla34_cnet test_dla34_cnet

    ./test_yolo3_coco4 &>> $out_file
    res_yolo3_coco4=$?
    print_output $res_yolo3_coco4 test_yolo3_coco4

done

echo "If errors occured, check logfile $out_file" 
