#!/bin/bash

function test_inference {
    ./test_$1
    ./test_rtinference $1_$2.rt 1 
    ./test_rtinference $1_$2.rt 4
}

sudo jeston_clock

# modes=( 1 ) # only FP32
# modes=( 1 2 ) # FP32 and FP16
modes=( 1 2 3 ) # FP32, FP16 and INT8

rm times_rtinference.csv
for i in "${modes[@]}"
do
    rm *rt
    if [ $i -eq 1 ]
    then
        export TKDNN_MODE=FP32
        mode=fp32
        echo -e "${ORANGE}Test FP32${NC}"
    fi
    if [ $i -eq 2 ]
    then
        export TKDNN_MODE=FP16
        mode=fp16
        echo -e "${ORANGE}Test FP16${NC}"
    fi
    if [ $i -eq 3 ]
    then
        export TKDNN_MODE=INT8
        export TKDNN_CALIB_LABEL_PATH=../demo/COCO_val2017/all_labels.txt
        export TKDNN_CALIB_IMG_PATH=../demo/COCO_val2017/all_images.txt
        mode=int8
        echo -e "${ORANGE}Test INT8${NC}"
        
	fi

    export TKDNN_BATCHSIZE=4
    echo -e "${ORANGE}Batch $TKDNN_BATCHSIZE ${NC}"
    
    test_inference yolo4_320 $mode
    test_inference yolo4 $mode
    test_inference yolo4_512 $mode
    test_inference yolo4_608 $mode
    test_inference yolo4tiny $mode
done



