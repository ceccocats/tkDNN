import argparse
import os
import msgpack
import lmdb
import random
import caffe
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAFFE WEIGHTS EXPORTER TO CUDNN')
    parser.add_argument('model',type=str,
        help='Path to prototxt network model')
    parser.add_argument('weights',type=str,
        help='Path to caffemodel file')
        
    parser.add_argument('--output', type=str, help="output directory", default="layers")
    parser.add_argument('--test_db', type=str, help="input db to test", default=None)

    args = parser.parse_args()


    print "\n\n ====== NET LOADED ====== "
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    n_lay = len(net.params)
    print "Number of layers: ", n_lay
    for i in xrange(n_lay):
        key = net.params.keys()[i]
        print "Layer", key
        print "    type: ", net.layer_dict[key].type 
        w = net.params[key][0].data
        b = net.params[key][1].data 
        print "    weights shape:", np.shape(w)
        print "    bias shape:", np.shape(b)
