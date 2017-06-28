import keras
from keras.models import load_model
import keras.backend.tensorflow_backend as KTF
import numpy as np
import argparse
import tensorflow as tf
import os
import msgpack
import lmdb
import random

def export_dense(name, weights, bias):
    print "########    EXPORT", name, "LAYER    ########"
    print "Original weighs:"
    print weights
    print bias, "\n"

    #input, filters
    I, C = np.shape(weights) 
    B = np.shape(bias)
    print "w shape: ", I, C
    print "b shape: ", B

    wgs = [ [ j[i] for j in weights ]  for i in xrange(C) ]
    wgs = np.array(wgs, dtype=np.float32)
    
    print "REPOSITIONED WEIGHTS:"
    print wgs

    bias = np.array(bias, dtype=np.float32)
    wgs.tofile(name + ".bin", format="f")
    bias.tofile(name + ".bias.bin", format="f")
    print "WEIGHTS saved\n"

def export_conv2d(name, weights, bias):
    print "########    EXPORT", name, "LAYER    ########"
    print "Original weighs:"
    print weights
    print bias, "\n"
  
    # height, width, input, filters
    H, W, N, C = np.shape(weights) 
    B = np.shape(bias)
    print "w shape: ", N, C, H, W
    print "b shape: ", B

    wgs = weights.transpose()
    wgs = wgs.transpose(0, 1, 3, 2)
    print "Final shape:", np.shape(wgs)    
    wgs = np.array(wgs.flatten(), dtype=np.float32)
    
    print "REPOSITIONED WEIGHTS:"
    print wgs

    bias = np.array(bias, dtype=np.float32)

    wgs.tofile(name + ".bin", format="f")
    bias.tofile(name + ".bias.bin", format="f")
    print "WEIGHTS saved\n"

def export_conv3d(name, weights, bias):
    print "########    EXPORT", name, "LAYER    ########"
    print "Original weighs:"
    print weights
    print bias, "\n"
  
    print np.shape(weights)
    # height, width, input, thickness, filters
    H, W, T, N, C = np.shape(weights) 
    B = np.shape(bias)
    print "w shape: ", T, C, H, W   #thickness is number of images for cudnn
    print "b shape: ", B

    wgs = weights.transpose()
    wgs = wgs.transpose(0, 1, 4, 3, 2)
    print "Final shape:", np.shape(wgs)    
    wgs = np.array(wgs.flatten(), dtype=np.float32)
    
    print "REPOSITIONED WEIGHTS:"
    print wgs

    bias = np.array(bias, dtype=np.float32)

    wgs.tofile(name + ".bin", format="f")
    bias.tofile(name + ".bias.bin", format="f")
    print "WEIGHTS saved\n"


def get_session(gpu_fraction=0.5):
    gpu_options = tf.GPUOptions(allow_growth=True) 
        #per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


#https://github.com/fchollet/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa
if __name__ == '__main__':
    KTF.set_session(get_session())

    parser = argparse.ArgumentParser(description='KERAS WEIGHTS EXPORTER TO CUDNN')
    parser.add_argument('model',type=str,
        help='Path to model h5 file. Model should be on the same path.')
    parser.add_argument('layers', type=str, help="layers list [ dense, conv2d ]", nargs='+')
    parser.add_argument('--output', type=str, help="output directory", default="layers")
    parser.add_argument('--test_db', type=str, help="input db to test", default=None)

    args = parser.parse_args()
    
    print "DATA FORMAT: ", keras.backend.image_data_format()
    
    print "Load model: ", args.model
    model = load_model(args.model)
    weights = model.get_weights()

    ws = np.shape(weights)
    print "Weights shape:", ws

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    num = 0
    name_num = 0
    for i in args.layers:
        if i == "conv3d":
            export_conv3d(args.output + "/conv" + str(name_num), weights[num], weights[num+1])
        elif i == "conv2d":
            export_conv2d(args.output + "/conv" + str(name_num), weights[num], weights[num+1])
        elif i == "dense":
            export_dense(args.output + "/dense" + str(name_num), weights[num], weights[num+1])
        else:
            print "error: ", i, "is not a layer type"
            break
        name_num += 1
        num += 2

    if args.test_db != None:
        print "Test on db: ", args.test_db
        db = lmdb.open(args.test_db, subdir=False, readonly=True, lock=False) 
        txn = db.begin()

        s = random.randint(0, txn.stat()["entries"]-1)
        print "camp number: ", s
        s = txn.get(str(s))
        c = msgpack.unpackb(s)
    
        print "Steer, throttle: ", c["actuators"]
        print "Speed (m/s): ", c["speed"]
        grid = np.asarray(c["bitmap"], np.float32)
        i = np.array(grid.flatten(), dtype=np.float32)
        i.tofile(args.output + "input.bin", format="f")
        X = grid[None, :, :]
    
        print "Prediction: ", model.predict(X)