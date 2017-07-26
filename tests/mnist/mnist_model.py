#!/usr/bin/env python
# mail:    admin@9crk.com
# author:  9crk.from China.ShenZhen
# time:    2017-03-22

import caffe
import numpy as np
import cv2
import sys
import Image
import matplotlib.pyplot as plt

model = 'lenet.prototxt';
weights = 'lenet.caffemodel';
net = caffe.Net(model,weights,caffe.TEST);
caffe.set_mode_gpu()
img = np.array(np.random.rand(28,28), dtype=np.float32)
#revert the image,and normalize it to 0-1 range

print "INPUT: ", img
img.tofile("input.bin", format="f")
print "SHAPE: ", np.shape(img)
out = net.forward_all(data=np.asarray([img]))

out = out[out.keys()[0]]
print out
print np.shape(out)
out.tofile("output.bin", format="f")
#print out['prob'][0]
#print out['prob'][0].argmax()


