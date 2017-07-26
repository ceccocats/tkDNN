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

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print "\n\n ====== NET LOADED ====== "
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    n_lay = len(net.params)
    print "Number of layers: ", n_lay
    for i in xrange(n_lay):
        key = net.params.keys()[i]
        print "Layer", key
        t = net.layer_dict[key].type 
        print "    type: ", t
        w = net.params[key][0].data
        b = net.params[key][1].data 
        print "    weights shape:", np.shape(w)
        print "    bias shape:", np.shape(b)
        
        w.tofile(args.output + "/" + t + str(i) + ".bin", format="f")
        b.tofile(args.output + "/" + t + str(i) + ".bias.bin", format="f")
