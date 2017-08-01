#!/bin/bash
if [ "$1" == "download" ]; then
    wget https://github.com/ceccocats/tkDNN/releases/download/testData/tkDNN_testwg.tar.gz --no-check-certificate
    tar -xf tkDNN_testwg.tar.gz
    rm tkDNN_testwg.tar.gz
    exit
fi

echo "build test Model"
cd test
python test_model.py
cd ..
cd mnist
python mnist_model.py
cd ..
echo "export weights"
python weights_exporter.py test/net.h5 --output test/layers
python caffe_weights_exporter.py mnist/lenet.prototxt mnist/lenet.caffemodel --output mnist/layers
