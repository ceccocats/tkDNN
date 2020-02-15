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


if __name__ == '__main__':


    print("DATA FORMAT: ", keras.backend.image_data_format())

    print("Load model: ", "ferrariS1.hdf5")
    model = load_model("ferrariS1.hdf5")
    model.summary()

    weights = model.get_weights()

    np.random.seed(2)
    x_angle = np.random.rand(1,100,4)
    x_gyro = np.random.rand(1,100,3)
    x_acc = np.random.rand(1,100,3)

    [yhat_delta_p, yhat_delta_q] = model.predict([x_angle, x_gyro, x_acc], batch_size=1, verbose=1)

    layer_name = 'bidirectional_3'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict([x_angle, x_gyro, x_acc])

    x_angle = np.array([x_angle])
    x_gyro = np.array([x_gyro])
    x_acc = np.array([x_acc])
    intermediate_output = np.array([intermediate_output])

    x_angle =  x_angle.transpose(0, 3, 1, 2)
    x_gyro =  x_gyro.transpose(0, 3, 1, 2)
    x_acc =  x_acc.transpose(0, 3, 1, 2)
    intermediate_output = intermediate_output.transpose(0, 3, 1, 2)

    print("x0: ", np.shape(x_angle))
    print("out: ",np.shape(intermediate_output))

    x_angle = np.array(x_angle.flatten(), dtype=np.float32)
    x_gyro = np.array(x_gyro.flatten(), dtype=np.float32)
    x_acc = np.array(x_acc.flatten(), dtype=np.float32)
    yhat_delta_p = np.array(yhat_delta_p.flatten(), dtype=np.float32)
    yhat_delta_q = np.array(yhat_delta_q.flatten(), dtype=np.float32)
    intermediate_output = np.array(intermediate_output.flatten(), dtype=np.float32)


    f = open("layers/input0.bin", mode='wb')
    bin_write(f, x_angle)
    f = open("layers/input1.bin", mode='wb')
    bin_write(f, x_gyro)
    f = open("layers/input2.bin", mode='wb')
    bin_write(f, x_acc)
    f = open("layers/output0.bin", mode='wb')
    bin_write(f, yhat_delta_p)
    f = open("layers/output1.bin", mode='wb')
    bin_write(f, yhat_delta_q)
    f = open("layers/output.bin", mode='wb')
    bin_write(f, intermediate_output)

