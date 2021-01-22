#!/bin/bash

cd build

RED='\033[1;31m'
GREEN='\033[1;32m'
ORANGE='\033[1;33m'
PINK='\033[1;95m'
NC='\033[0m' # No Color

function print_output {
    if [ $1 -eq 0 ]; then
        echo -e "$2 ${GREEN}OK${NC}"
    elif [ $1 -eq 1 ]; then
        echo -e "$2 ${RED}FATAL ERROR${NC}"
    elif [ $1 -eq 2 ] || [ $1 -eq 10 ]; then
        echo -e "$2 ${PINK}CUDNN ERROR${NC}"
    elif [ $1 -eq 4 ] || [ $1 -eq 12 ]; then
        echo -e "$2 ${PINK}TENSORRT ERROR${NC}"
    elif [ $1 -eq 8 ]; then
        echo -e "$2 ${PINK}CUDNN vs TENSORRT ERROR${NC}"
    elif [ $1 -eq 6 ]; then
        echo -e "$2 ${PINK}CUDNN & TENSORTRT ERROR${NC}"
    elif [ $1 -eq 14 ]; then
        echo -e "$2 ${PINK}ERROR FOR EVERY CHECK${NC}"
    else
        echo -e "$2 ${RED}NOT OKAY (OPENCV maybe)${NC}"
    fi 

} 

out_file=results.log
rm $out_file

function test_net {
    ./test_$1 &>> $out_file
    print_output $? $1
    ./test_rtinference $1*.rt $TKDNN_BATCHSIZE &>> $out_file
    print_output $? "batched $1"
}


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

    export TKDNN_BATCHSIZE=2
    echo -e "${ORANGE}Batch $TKDNN_BATCHSIZE ${NC}"
    
    test_net mnist    
    # ./test_imuodom &>> $out_file
    # print_output $? imuodom

    test_net yolo4
    test_net yolo4-csp
    test_net yolo4x
    test_net yolo4_berkeley
    test_net yolo4tiny
    test_net yolo3
    test_net yolo3_berkeley
    test_net yolo3_coco4
    test_net yolo3_flir
    test_net yolo3_512
    test_net yolo3tiny
    test_net yolo3tiny_512
    test_net yolo2
    test_net yolo2_voc
    #test_net yolo2tiny
    test_net csresnext50-panet-spp
    #test_net csresnext50-panet-spp_berkeley
    test_net resnet101_cnet
    test_net dla34_cnet
    test_net mobilenetv2ssd
    test_net mobilenetv2ssd512
    test_net bdd-mobilenetv2ssd
done

echo "If errors occured, check logfile $out_file" 
