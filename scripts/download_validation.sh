#!/bin/bash

cd demo
wget https://cloud.hipert.unimore.it/s/LNxBDk4wzqXPL8c/download -O COCO_val2017.zip
unzip -d COCO_val2017 COCO_val2017.zip
rm COCO_val2017.zip
cd COCO_val2017/
realpath labels/* > all_labels.txt
