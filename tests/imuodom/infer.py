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
import pickle

def bin_write(f, data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)

# USE weight_exporter to generare wgs bins
if __name__ == '__main__':
    

    print("DATA FORMAT: ", keras.backend.image_data_format())

    print("Load model: ", "ferrariSEP.hdf5")
    model = load_model("ferrariSEP.hdf5")
    model.summary()
    weights = model.get_weights()

    indata = pickle.load(open("input.pk", 'rb'))
    outdata = pickle.load(open("output.pk", 'rb'))

    x_angle = indata[0]
    x_gyro = indata[1]
    x_acc = indata[2]
    
    [yhat_delta_p, yhat_delta_q] = model.predict(indata, batch_size=1, verbose=1)
    predictdata = [yhat_delta_p, yhat_delta_q]

    error = outdata[0] - predictdata[0]
    print("error delta_p: ", error.sum())
    error = outdata[1] - predictdata[1]
    print("error delta_q: ", error.sum())


    #layer_name = 'dense_4'
    #intermediate_layer_model = Model(inputs=model.input,
    #                                 outputs=model.get_layer(layer_name).output)
    #intermediate_output = intermediate_layer_model.predict([x_angle, x_gyro, x_acc])


    x_angle = np.array([x_angle])
    x_gyro = np.array([x_gyro])
    x_acc = np.array([x_acc])
    #intermediate_output = np.array([intermediate_output])

    x_angle =  x_angle.transpose(1, 3, 0, 2)
    x_gyro =  x_gyro.transpose(1, 3, 0, 2)
    x_acc =  x_acc.transpose(1, 3, 0, 2)
    #intermediate_output = intermediate_output.transpose(0, 3, 1, 2)
    #print("Aggregate:")
    #print(intermediate_output.tolist())
 
    print("x0: ", np.shape(x_angle))
    #print("out: ",np.shape(intermediate_output))

    x_angle = np.array(x_angle.flatten(), dtype=np.float32)
    x_gyro = np.array(x_gyro.flatten(), dtype=np.float32)
    x_acc = np.array(x_acc.flatten(), dtype=np.float32)
    yhat_delta_p = np.array(yhat_delta_p.flatten(), dtype=np.float32)
    yhat_delta_q = np.array(yhat_delta_q.flatten(), dtype=np.float32)
    #intermediate_output = np.array(intermediate_output.flatten(), dtype=np.float32)


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
    #f = open("layers/output.bin", mode='wb')
    #bin_write(f, intermediate_output)

