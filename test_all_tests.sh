#!/bin/bash

cd build

RED='\033[1;31m'
GREEN='\033[1;32m'
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

echo "If errors occured, check logfile $out_file" 
