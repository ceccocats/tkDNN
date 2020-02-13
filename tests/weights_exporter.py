import keras
from keras.models import load_model
import keras.backend.tensorflow_backend as KTF
import numpy as np
import argparse
import tensorflow as tf
import os
import random
import struct
from keras.models import Sequential, Model

def bin_write(f, data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

def export_layer(name, weights, bias):
    print ("########    EXPORT", name, "LAYER    ########")

    print("wgs pretranpose: ", np.shape(weights))
    # convert NHWC to NCHW
    if(weights.ndim == 4):
        weights = weights.transpose(3,2,0,1)
    elif(weights.ndim == 3):
        weights = weights.transpose(2,1,0)
    else:
        print("Ndim", weights.ndim)
        raise("not implemented with dim" )

    print("weights: ", np.shape(weights))
    print("bias: ", np.shape(bias))

    weights = np.array(weights.flatten(), dtype=np.float32)
    bias = np.array(bias, dtype=np.float32)
    print(len(weights) + len(bias))

    f = open(name + ".bin", mode='wb')
    bin_write(f, weights)
    bin_write(f, bias)
    print ("WEIGHTS saved\n")

def export_bidir(name, weights):
    print ("########    EXPORT", name, "LAYER    ########")
 
    for w in weights:
        print(np.shape(w))


#https://github.com/fchollet/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa
if __name__ == '__main__':
    print("DATA FORMAT: ", keras.backend.image_data_format())

    parser = argparse.ArgumentParser(description='KERAS WEIGHTS EXPORTER TO CUDNN')
    parser.add_argument('model',type=str,
        help='Path to model h5 file. Model should be on the same path.')
    parser.add_argument('--output', type=str, help="output directory", default="layers")

    args = parser.parse_args()
    
    print("DATA FORMAT: ", keras.backend.image_data_format())
    
    print("Load model: ", args.model)
    model = load_model(args.model)
    model.summary()

    weights = model.get_weights()

    ws = np.shape(weights)
    print("Weights shape:", ws)

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    name_num = 0
    for l in model.layers:
        print("\n\nNAME: ", l.name)
        print("input: ", l.input_shape, " output: ", l.output_shape)
        wgs = l.get_weights()
        print("wgs num: ", len(wgs))

        name = l.name 
        if name.startswith("conv3d"):
            export_layer(args.output + "/" + name, wgs[0], wgs[1])
        elif name.startswith("conv2d"):
            export_layer(args.output + "/" + name, wgs[0], wgs[1])
        elif name.startswith("conv1d"):
            export_layer(args.output + "/" + name, wgs[0], wgs[1])     
        elif name.startswith("dense"):
            export_layer(args.output + "/" + name, wgs[0], wgs[1])
        elif name.startswith("bidirectional"):
            export_bidir(args.output + "/" + name, wgs)
        else:
            print ("skip:", name, "has no weights")
            continue


